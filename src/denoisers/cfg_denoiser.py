from torch import FloatTensor, BoolTensor
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .denoiser_proto import Denoiser

@dataclass
class CFGDenoiser:
  denoiser: Denoiser
  cross_attention_conds: FloatTensor
  cfg_scale: float = 5.
  added_cond_kwargs: Dict[str, Any] = {}
  cross_attention_mask: Optional[BoolTensor] = None

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    conds_per_sample = 2
    noised_latents_in: FloatTensor = noised_latents.repeat(conds_per_sample, 1, 1, 1)
    del noised_latents
    sigma_in: FloatTensor = sigma.repeat(conds_per_sample)
    del sigma
    denoised_latents: FloatTensor = self.denoiser.forward(
      input=noised_latents_in,
      sigma=sigma_in,
      encoder_hidden_states=self.cross_attention_conds,
      cross_attention_mask=self.cross_attention_mask,
      added_cond_kwargs=self.added_cond_kwargs,
    )
    uncond, cond = denoised_latents.chunk(conds_per_sample)
    del noised_latents_in, sigma_in
    return uncond + (cond - uncond) * self.cfg_scale