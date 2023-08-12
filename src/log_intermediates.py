from functools import partial
from os.path import join
from typing import Callable, TypedDict, List
from typing_extensions import TypeAlias
from torch import Tensor
from PIL import Image
from .latents_to_pils import LatentsToPils

class KSamplerCallbackPayload(TypedDict):
  x: Tensor
  i: int
  sigma: Tensor
  sigma_hat: Tensor
  denoised: Tensor

KSamplerCallback: TypeAlias = Callable[[KSamplerCallbackPayload], None]

def log_intermediate(
  latents_to_pils: LatentsToPils,
  intermediates_paths: List[str],
  payload: KSamplerCallbackPayload,
) -> None:
  sample_pils: List[Image.Image] = latents_to_pils(payload['denoised'])
  for intermediates_path, img in zip(intermediates_paths, sample_pils):
    fname: str = f"{payload['i']:03d}_{payload['sigma'].item():.4f}.jpg"
    fpath: str = join(intermediates_path, fname)
    img.save(fpath, subsampling=0, quality=95)

LogIntermediates: TypeAlias = Callable[[KSamplerCallbackPayload], None]
LogIntermediatesFactory: TypeAlias = Callable[[List[str]], LogIntermediates]
def make_log_intermediates_factory(latents_to_pils: LatentsToPils) -> LogIntermediatesFactory:
  return lambda intermediates_paths: partial(log_intermediate, latents_to_pils, intermediates_paths)