# FLUX.1-dev RunPod Serverless Text-to-Image API

A production-oriented Serverless text-to-image API built with
[RunPod Serverless](https://www.runpod.io/serverless-gpu),
[Black Forest Labs FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev),
Hugging Face Diffusers, PyTorch, and Docker.

The service accepts a text prompt and generation parameters, performs FLUX.1-dev
inference on a RunPod GPU worker, and returns the generated JPEG as Base64 JSON.

## Architecture

```text
Client
  |
  | POST /run
  v
RunPod Serverless Queue
  |
  v
Docker Worker
  |
  +--> handler.py
  |      |
  |      +--> request validation (schemas.py)
  |      |
  |      +--> FluxModel singleton (model.py)
  |               |
  |               +--> RunPod cached FLUX.1-dev snapshot
  |               |
  |               +--> BF16 CUDA inference
  |
  +--> JPEG encoding
  |
  v
Base64 JSON response
```

## Key Design Decisions

### RunPod model caching

FLUX.1-dev is too large to download reliably into the worker's small ephemeral
filesystem during every cold start.

The initial deployment exposed this constraint with:

```text
IO Error: No space left on device (os error 28)
```

The production version therefore uses RunPod's Hugging Face model cache.
The worker detects the mounted FLUX snapshot under
`/runpod-volume/huggingface-cache/hub` and loads directly from it.

This avoids repeatedly downloading the model into ephemeral worker storage and
significantly improves cold-start reliability.

### Model lifecycle

`FluxModel` is instantiated once at module scope. The FLUX pipeline is loaded
lazily on the first request and remains resident for subsequent requests handled
by the same worker.

This avoids reloading the model for every generation.

### GPU execution

The pipeline uses:

- PyTorch
- Hugging Face Diffusers `FluxPipeline`
- `torch.bfloat16`
- CUDA
- One request per GPU worker

The deployed endpoint uses RunPod's 48 GB GPU tier with A40 and RTX A6000
enabled.

### API output

Generated images are converted to JPEG and Base64 encoded.

This keeps the API self-contained and works comfortably within RunPod payload
limits for the tested 1024x1024 output.

## Request

Example:

```json
{
  "input": {
    "prompt": "A cinematic photograph of a futuristic GPU data center at night, rows of illuminated servers, dramatic volumetric lighting, highly detailed",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  }
}
```

The same request is available in `examples/request.json`.

## Response

The handler returns:

```json
{
  "image": {
    "encoding": "base64",
    "format": "jpeg",
    "data": "<base64 JPEG>"
  },
  "parameters": {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42
  },
  "metrics": {
    "inference_time_seconds": 27.63,
    "total_time_seconds": 27.63
  }
}
```

## Example Output

![FLUX.1-dev generated GPU data center](examples/output.jpg)

Test configuration:

| Parameter | Value |
| --- | --- |
| Model | `black-forest-labs/FLUX.1-dev` |
| Resolution | 1024x1024 |
| Steps | 28 |
| Guidance scale | 3.5 |
| Seed | 42 |
| GPU tier | 48 GB |
| GPU types | A40 / RTX A6000 |

## Performance

A controlled cold/warm test was performed using the same prompt, parameters,
seed, endpoint, and worker.

| Metric | Cold request | Warm request |
| --- | ---: | ---: |
| RunPod queue/delay | 11.73 s | 0.124 s |
| RunPod execution | 35.17 s | 27.81 s |
| Approx. end-to-end | 46.90 s | 27.93 s |
| FLUX inference | 27.75 s | 27.63 s |
| Model initialization | 6.31 s | Reused |

The warm request reused the same worker and already-loaded pipeline. This reduced
approximate end-to-end latency from 46.90 seconds to 27.93 seconds, roughly a
40% improvement.

## RunPod Configuration

The validated endpoint configuration is:

| Setting | Value |
| --- | --- |
| Endpoint type | Queue based |
| GPU memory | 48 GB |
| Enabled GPUs | A40, RTX A6000 |
| GPU count | 1 |
| Max workers | 1 |
| Active workers | 0 |
| Idle timeout | 30 seconds |
| Execution timeout | 600 seconds |
| Autoscaling | Queue delay |
| Queue delay | 4 seconds |
| Model | `black-forest-labs/FLUX.1-dev` |
| Model delivery | RunPod cached model |
| Hugging Face authentication | RunPod Secret |

## Local Development

Python 3.11 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Install PyTorch appropriate for the local CUDA environment, then install the
application dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -v
```

The current test suite contains 12 tests covering request validation, model
lifecycle behavior, handler output encoding, and error propagation.

## Docker

Build the production worker for RunPod's Linux AMD64 environment:

```bash
docker build --platform linux/amd64 \
  -t runpod-flux-serverless:v0.1.1 .
```

Local GPU smoke test:

```bash
docker run --rm --gpus all \
  --entrypoint python \
  runpod-flux-serverless:v0.1.1 \
  -c "import torch, model; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('Model loaded:', model.flux_model.pipeline is not None)"
```

The model should report `Model loaded: False` during this test because loading is
lazy and occurs when the first inference request arrives.

## Deployment

The validated Docker image is:

```text
jorgelhuerta1020/runpod-flux-serverless:v0.1.1
```

Validated image digest:

```text
sha256:c6dca87b0f359a2c1c01f474a259a0d9b40d7dadecd038a014ff9bf65b436dca
```

Deployment workflow:

1. Build and push the Linux AMD64 Docker image.
2. Create a RunPod queue-based Serverless endpoint.
3. Select the 48 GB GPU tier.
4. Configure FLUX.1-dev as the RunPod cached Hugging Face model.
5. Store the Hugging Face access token as a RunPod Secret.
6. Configure worker scaling and execution timeout.
7. Deploy the versioned Docker image.
8. Submit jobs through RunPod `/run`.
9. Poll `/status/{job_id}` until completion.

## Project Structure

```text
.
├── Dockerfile
├── handler.py
├── model.py
├── schemas.py
├── requirements.txt
├── pytest.ini
├── examples/
│   ├── request.json
│   ├── response.json
│   └── output.jpg
└── tests/
    └── test_local.py
```

## Security

Secrets are never embedded in the Docker image or committed to Git.

The gated Hugging Face token is provided to RunPod using its Secrets mechanism.

Prompts are also not written to application logs.

## Failure Handling

Input validation errors return structured JSON responses.

Unexpected inference/runtime failures are raised from the handler so RunPod can
correctly mark the job as failed rather than returning a misleading successful
response.

## Model License

FLUX.1-dev is a gated model distributed by Black Forest Labs under its applicable
FLUX.1-dev license. Users deploying this project are responsible for ensuring
their intended use complies with the model license.

## Technologies

- Python 3.11
- RunPod Serverless
- FLUX.1-dev
- Hugging Face Diffusers
- PyTorch
- CUDA
- Docker
- Pillow
- Pytest