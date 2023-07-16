from torch import FloatTensor, stack
from typing import List, Optional
from .get_embed import GetEmbed
from ..sample_spec.sample_spec import SampleSpec

def embed_batch(
  batch_frames: List[SampleSpec],
  get_cond_embed: GetEmbed,
  get_uncond_embed: Optional[GetEmbed],
) -> FloatTensor:
  unconds: List[FloatTensor] = [] if get_uncond_embed is None else [
    get_uncond_embed(sample_spec=frame) for frame in batch_frames
  ]
  conds: List[FloatTensor] = [
    get_cond_embed(sample_spec=frame) for frame in batch_frames
  ]
  return stack([*unconds, *conds])