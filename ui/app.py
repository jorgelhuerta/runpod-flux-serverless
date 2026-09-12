import base64
import io
import os
import time

import requests
import streamlit as st
from PIL import Image


RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

RUNPOD_BASE_URL = (
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
    if RUNPOD_ENDPOINT_ID
    else None
)

POLL_INTERVAL_SECONDS = 1.0
MAX_WAIT_SECONDS = 600


st.set_page_config(
    page_title="FLUX.1-dev Image Generator",
    page_icon="✨",
    layout="wide",
)


def validate_configuration() -> None:
    if not RUNPOD_ENDPOINT_ID:
        st.error("RUNPOD_ENDPOINT_ID is not configured.")
        st.stop()

    if not RUNPOD_API_KEY:
        st.error("RUNPOD_API_KEY is not configured.")
        st.stop()


def auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def submit_job(payload: dict) -> str:
    response = requests.post(
        f"{RUNPOD_BASE_URL}/run",
        headers=auth_headers(),
        json={"input": payload},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    job_id = data.get("id")
    if not job_id:
        raise RuntimeError(f"RunPod did not return a job ID: {data}")

    return job_id


def wait_for_job(job_id: str) -> dict:
    started = time.perf_counter()

    while True:
        if time.perf_counter() - started > MAX_WAIT_SECONDS:
            raise TimeoutError(
                f"Generation exceeded {MAX_WAIT_SECONDS} seconds."
            )

        response = requests.get(
            f"{RUNPOD_BASE_URL}/status/{job_id}",
            headers=auth_headers(),
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        status = result.get("status", "UNKNOWN")

        if status == "COMPLETED":
            return result

        if status in {
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
        }:
            error = result.get("error", "Unknown RunPod error")
            raise RuntimeError(f"{status}: {error}")

        time.sleep(POLL_INTERVAL_SECONDS)


def decode_image(output: dict) -> tuple[Image.Image, bytes]:
    image_data = output.get("image", {}).get("data")

    if not image_data:
        raise RuntimeError("Response did not contain image data.")

    image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    return image, image_bytes


validate_configuration()


st.title("FLUX.1-dev Image Generator")

st.caption(
    "RunPod Serverless • FLUX.1-dev • 48 GB GPU • "
    "Hugging Face cached model"
)

st.divider()


left, right = st.columns([1, 1.15], gap="large")


with left:
    st.subheader("Generate an image")

    prompt = st.text_area(
        "Prompt",
        value=(
            "A cinematic photograph of a futuristic GPU data center "
            "at night, rows of illuminated servers, dramatic volumetric "
            "lighting, highly detailed"
        ),
        height=150,
    )

    col1, col2 = st.columns(2)

    with col1:
        width = st.selectbox(
            "Width",
            [512, 768, 1024],
            index=2,
        )

        steps = st.slider(
            "Inference steps",
            min_value=10,
            max_value=50,
            value=28,
        )

    with col2:
        height = st.selectbox(
            "Height",
            [512, 768, 1024],
            index=2,
        )

        guidance_scale = st.slider(
            "Guidance scale",
            min_value=1.0,
            max_value=7.0,
            value=3.5,
            step=0.1,
        )

    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
    )

    generate = st.button(
        "Generate image",
        type="primary",
        use_container_width=True,
    )


with right:
    st.subheader("Result")

    result_container = st.container()

    if not generate:
        result_container.info(
            "Enter a prompt and click Generate image."
        )


if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    request_payload = {
        "prompt": prompt.strip(),
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": int(seed),
    }

    overall_started = time.perf_counter()

    try:
        with result_container:
            status = st.status(
                "Submitting request to RunPod...",
                expanded=True,
            )

            job_id = submit_job(request_payload)

            status.write(f"Job ID: `{job_id}`")
            status.update(
                label="Generating image...",
                state="running",
            )

            result = wait_for_job(job_id)

            elapsed = time.perf_counter() - overall_started

            output = result.get("output")

            if not isinstance(output, dict):
                raise RuntimeError(
                    f"Invalid endpoint response: {result}"
                )

            image, image_bytes = decode_image(output)

            status.update(
                label="Generation complete",
                state="complete",
                expanded=False,
            )

            st.image(
                image,
                caption=prompt,
                use_container_width=True,
            )

            metrics = output.get("metrics", {})
            parameters = output.get("parameters", {})

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Inference",
                f"{metrics.get('inference_time_seconds', 0):.2f}s",
            )

            m2.metric(
                "End-to-end",
                f"{elapsed:.2f}s",
            )

            m3.metric(
                "Seed",
                parameters.get("seed", seed),
            )

            st.download_button(
                "Download JPEG",
                data=image_bytes,
                file_name=f"flux-{job_id}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

            with st.expander("Generation details"):
                st.json(
                    {
                        "job_id": job_id,
                        "parameters": parameters,
                        "metrics": metrics,
                        "runpod_delay_time_ms": result.get(
                            "delayTime"
                        ),
                        "runpod_execution_time_ms": result.get(
                            "executionTime"
                        ),
                    }
                )

    except requests.HTTPError as exc:
        st.error(
            f"RunPod API request failed: "
            f"{exc.response.status_code} "
            f"{exc.response.text}"
        )

    except Exception as exc:
        st.error(f"Generation failed: {exc}")


st.divider()

st.caption(
    "The API key is kept server-side and is never exposed "
    "to the browser."
)