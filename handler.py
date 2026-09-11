import base64
import io
import logging
import time
from typing import Any, Dict

import runpod

from model import flux_model
from schemas import ValidationError, parse_generation_request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def encode_image_to_base64(image) -> str:
    buffer = io.BytesIO()

    # FLUX normally returns RGB, but this keeps JPEG serialization safe.
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(
        buffer,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    request_started = time.perf_counter()

    try:
        job_input = job.get("input")

        request = parse_generation_request(job_input)

    except ValidationError as exc:
        logger.warning("Invalid request: %s", exc)

        return {
            "error": {
                "type": "validation_error",
                "message": str(exc),
            }
        }

    logger.info(
        "Received generation request: %dx%d, steps=%d",
        request.width,
        request.height,
        request.num_inference_steps,
    )

    try:
        result = flux_model.generate(request)

        image_base64 = encode_image_to_base64(result.image)

        total_time = time.perf_counter() - request_started

        return {
            "image": {
                "encoding": "base64",
                "format": "jpeg",
                "data": image_base64,
            },
            "parameters": {
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": result.seed,
            },
            "metrics": {
                "inference_time_seconds": round(
                    result.inference_time_seconds,
                    3,
                ),
                "total_time_seconds": round(
                    total_time,
                    3,
                ),
            },
        }

    except Exception:
        logger.exception("Image generation failed")
        raise


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
        }
    )