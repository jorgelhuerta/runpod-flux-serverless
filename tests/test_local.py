import pytest

import base64
import io

from PIL import Image

import handler as handler_module
from model import GenerationResult

from schemas import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    ValidationError,
    parse_generation_request,
)


def test_minimal_valid_request():
    request = parse_generation_request(
        {
            "prompt": "A futuristic city at sunset"
        }
    )

    assert request.prompt == "A futuristic city at sunset"
    assert request.width == DEFAULT_WIDTH
    assert request.height == DEFAULT_HEIGHT
    assert request.num_inference_steps == DEFAULT_STEPS
    assert request.guidance_scale == DEFAULT_GUIDANCE_SCALE


def test_custom_valid_request():
    request = parse_generation_request(
        {
            "prompt": "An astronaut walking through a neon city",
            "width": 768,
            "height": 768,
            "num_inference_steps": 30,
            "guidance_scale": 4.0,
            "seed": 42,
        }
    )

    assert request.width == 768
    assert request.height == 768
    assert request.num_inference_steps == 30
    assert request.guidance_scale == 4.0
    assert request.seed == 42


def test_missing_prompt():
    with pytest.raises(ValidationError):
        parse_generation_request({})


def test_empty_prompt():
    with pytest.raises(ValidationError):
        parse_generation_request(
            {
                "prompt": "   "
            }
        )


def test_invalid_dimensions():
    with pytest.raises(ValidationError):
        parse_generation_request(
            {
                "prompt": "test",
                "width": 4096,
            }
        )


def test_invalid_steps():
    with pytest.raises(ValidationError):
        parse_generation_request(
            {
                "prompt": "test",
                "num_inference_steps": 100,
            }
        )


def test_invalid_seed():
    with pytest.raises(ValidationError):
        parse_generation_request(
            {
                "prompt": "test",
                "seed": -1,
            }
        )


        

from model import FluxModel


def test_model_starts_unloaded():
    model = FluxModel()

    assert model.pipeline is None
    assert model.model_load_time_seconds is None


def test_model_rejects_missing_cuda(monkeypatch):
    model = FluxModel()

    monkeypatch.setattr(
        "model.torch.cuda.is_available",
        lambda: False,
    )

    try:
        model.load()
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "CUDA GPU is required" in str(exc)


def test_handler_rejects_invalid_request():
    response = handler_module.handler(
        {
            "input": {}
        }
    )

    assert response["error"]["type"] == "validation_error"
    assert "prompt" in response["error"]["message"]


def test_handler_generates_encoded_image(monkeypatch):
    test_image = Image.new(
        "RGB",
        (512, 512),
        "white",
    )

    def fake_generate(request):
        return GenerationResult(
            image=test_image,
            seed=42,
            inference_time_seconds=1.25,
        )

    monkeypatch.setattr(
        handler_module.flux_model,
        "generate",
        fake_generate,
    )

    response = handler_module.handler(
        {
            "input": {
                "prompt": "A futuristic city at sunset",
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "seed": 42,
            }
        }
    )

    assert response["image"]["encoding"] == "base64"
    assert response["image"]["format"] == "jpeg"

    assert response["parameters"]["seed"] == 42
    assert response["parameters"]["width"] == 512
    assert response["parameters"]["height"] == 512

    image_bytes = base64.b64decode(
        response["image"]["data"]
    )

    decoded_image = Image.open(
        io.BytesIO(image_bytes)
    )

    assert decoded_image.format == "JPEG"
    assert decoded_image.size == (512, 512)


def test_handler_propagates_generation_failure(monkeypatch):
    def fake_generate(request):
        raise RuntimeError("simulated GPU failure")

    monkeypatch.setattr(
        handler_module.flux_model,
        "generate",
        fake_generate,
    )

    try:
        handler_module.handler(
            {
                "input": {
                    "prompt": "test"
                }
            }
        )

        assert False, "Expected RuntimeError"

    except RuntimeError as exc:
        assert "simulated GPU failure" in str(exc)