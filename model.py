import secrets
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import FluxPipeline
from PIL import Image

from schemas import GenerationRequest

from pathlib import Path

HF_TOKEN = os.getenv("HF_TOKEN")


logger = logging.getLogger(__name__)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "black-forest-labs/FLUX.1-dev",
)

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")

HF_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")


def resolve_cached_model_path(model_id: str) -> Optional[str]:
    if "/" not in model_id:
        return None

    org, name = model_id.split("/", 1)
    model_root = HF_CACHE_ROOT / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        snapshot = snapshots_dir / revision

        if snapshot.is_dir():
            return str(snapshot)

    if snapshots_dir.is_dir():
        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshots:
            return str(snapshots[0])

    return None


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    seed: int
    inference_time_seconds: float


class FluxModel:
    def __init__(self) -> None:
        self.pipeline: Optional[FluxPipeline] = None
        self.model_load_time_seconds: Optional[float] = None

    def load(self) -> None:
        if self.pipeline is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required to load FLUX.1-dev."
            )

        cached_path = resolve_cached_model_path(MODEL_ID)

        source = (
            LOCAL_MODEL_PATH
            or cached_path
            or MODEL_ID
        )

        if LOCAL_MODEL_PATH:
            logger.info("Using explicit local model path: %s", source)
        elif cached_path:
            logger.info("Using RunPod cached model: %s", source)
        else:
            logger.info("Using Hugging Face model ID: %s", source)

            logger.info("Loading FLUX model from %s", source)

        started = time.perf_counter()

        pipeline = FluxPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
        )

        pipeline.to("cuda")

        self.model_load_time_seconds = (
            time.perf_counter() - started
        )

        self.pipeline = pipeline

        logger.info(
            "FLUX model loaded in %.2f seconds",
            self.model_load_time_seconds,
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        if self.pipeline is None:
            self.load()

        assert self.pipeline is not None

        seed = (
            request.seed
            if request.seed is not None
            else secrets.randbits(63)
        )

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(seed)

        logger.info(
            "Generating image: %dx%d, steps=%d, seed=%d",
            request.width,
            request.height,
            request.num_inference_steps,
            seed,
        )

        started = time.perf_counter()

        output = self.pipeline(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            generator=generator,
        )

        inference_time = time.perf_counter() - started

        image = output.images[0]

        logger.info(
            "Image generated in %.2f seconds",
            inference_time,
        )

        return GenerationResult(
            image=image,
            seed=seed,
            inference_time_seconds=inference_time,
        )


flux_model = FluxModel()