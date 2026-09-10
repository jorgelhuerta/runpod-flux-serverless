from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_MAX_SEQUENCE_LENGTH = 512

MIN_DIMENSION = 512
MAX_DIMENSION = 1024
MAX_STEPS = 50
MAX_PROMPT_LENGTH = 1000


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_inference_steps: int = DEFAULT_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: Optional[int] = None
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH


class ValidationError(ValueError):
    """Raised when the endpoint receives invalid user input."""


def _validate_dimension(name: str, value: Any) -> int:
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer.")

    if value < MIN_DIMENSION or value > MAX_DIMENSION:
        raise ValidationError(
            f"{name} must be between {MIN_DIMENSION} and {MAX_DIMENSION}."
        )

    if value % 16 != 0:
        raise ValidationError(f"{name} must be divisible by 16.")

    return value


def parse_generation_request(job_input: Dict[str, Any]) -> GenerationRequest:
    if not isinstance(job_input, dict):
        raise ValidationError("input must be a JSON object.")

    prompt = job_input.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("prompt is required and must be a non-empty string.")

    prompt = prompt.strip()

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"prompt must not exceed {MAX_PROMPT_LENGTH} characters."
        )

    width = _validate_dimension(
        "width",
        job_input.get("width", DEFAULT_WIDTH),
    )

    height = _validate_dimension(
        "height",
        job_input.get("height", DEFAULT_HEIGHT),
    )

    steps = job_input.get(
        "num_inference_steps",
        DEFAULT_STEPS,
    )

    if not isinstance(steps, int):
        raise ValidationError("num_inference_steps must be an integer.")

    if not 1 <= steps <= MAX_STEPS:
        raise ValidationError(
            f"num_inference_steps must be between 1 and {MAX_STEPS}."
        )

    guidance_scale = job_input.get(
        "guidance_scale",
        DEFAULT_GUIDANCE_SCALE,
    )

    if not isinstance(guidance_scale, (int, float)):
        raise ValidationError("guidance_scale must be a number.")

    guidance_scale = float(guidance_scale)

    if not 0.0 <= guidance_scale <= 10.0:
        raise ValidationError(
            "guidance_scale must be between 0.0 and 10.0."
        )

    seed = job_input.get("seed")

    if seed is not None:
        if not isinstance(seed, int):
            raise ValidationError("seed must be an integer.")

        if seed < 0:
            raise ValidationError("seed must be greater than or equal to 0.")

    max_sequence_length = job_input.get(
        "max_sequence_length",
        DEFAULT_MAX_SEQUENCE_LENGTH,
    )

    if not isinstance(max_sequence_length, int):
        raise ValidationError("max_sequence_length must be an integer.")

    if not 1 <= max_sequence_length <= 512:
        raise ValidationError(
            "max_sequence_length must be between 1 and 512."
        )

    return GenerationRequest(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        max_sequence_length=max_sequence_length,
    )