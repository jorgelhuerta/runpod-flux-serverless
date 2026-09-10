import pytest

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