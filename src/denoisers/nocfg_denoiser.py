from .denoiser_proto import Denoiser
from torch import FloatTensor, BoolTensor
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class NoCFGDenoiser:
  denoiser: Denoiser
  cross_attention_conds: FloatTensor
  added_cond_kwargs: Dict[str, Any] = field(default_factory={})
  cross_attention_mask: Optional[BoolTensor] = None

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    return self.denoiser.forward(
      input=noised_latents,
      sigma=sigma,
      encoder_hidden_states=self.cross_attention_conds,
      cross_attention_mask=self.cross_attention_mask,
      added_cond_kwargs=self.added_cond_kwargs,
    )