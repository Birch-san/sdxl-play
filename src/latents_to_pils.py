from functools import partial
from typing import List, Callable
from typing_extensions import TypeAlias
from PIL import Image
from torch import Tensor, no_grad
from diffusers.models import AutoencoderKL

LatentsToBCHW: TypeAlias = Callable[[Tensor], Tensor]
LatentsToPils: TypeAlias = Callable[[Tensor], List[Image.Image]]

@no_grad()
def latents_to_bchw(vae: AutoencoderKL, latents: Tensor) -> Tensor:
  latents: Tensor = 1 / vae.config.scaling_factor * latents

  images: Tensor = vae.decode(latents.to(vae.dtype)).sample

  images: Tensor = (images / 2 + 0.5).clamp(0, 1)
  return images

def make_latents_to_bchw(vae: AutoencoderKL) -> LatentsToPils:
  return partial(latents_to_bchw, vae)

def latents_to_pils(latents_to_bchw: LatentsToBCHW, latents: Tensor) -> List[Image.Image]:
  images: Tensor = latents_to_bchw(latents)

  # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
  images = images.cpu().permute(0, 2, 3, 1).float().numpy()
  images = (images * 255).round().astype("uint8")

  pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
  return pil_images

def make_latents_to_pils(latents_to_bchw: LatentsToBCHW) -> LatentsToPils:
  return partial(latents_to_pils, latents_to_bchw)