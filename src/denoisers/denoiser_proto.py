from typing import Protocol, Dict, Optional, Any
from torch import BoolTensor, FloatTensor

class Denoiser(Protocol):
  sigma_min: FloatTensor
  sigma_max: FloatTensor
  sigmas: FloatTensor
  def forward(
    self,
    input: FloatTensor,
    sigma: FloatTensor,
    encoder_hidden_states: FloatTensor,
    added_cond_kwargs: Dict[str, Any] = {},
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> FloatTensor: ...