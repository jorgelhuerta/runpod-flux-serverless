import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import FluxPipeline
from PIL import Image

from schemas import GenerationRequest


logger = logging.getLogger(__name__)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "black-forest-labs/FLUX.1-dev",
)

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")


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

        source = LOCAL_MODEL_PATH or MODEL_ID

        logger.info("Loading FLUX model from %s", source)

        started = time.perf_counter()

        pipeline = FluxPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
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
            else torch.seed()
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