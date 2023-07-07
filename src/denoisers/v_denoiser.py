from k_diffusion.external import DiscreteVDDPMDenoiser
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
import torch
from torch import Tensor, BoolTensor, FloatTensor
from typing import Union, Optional, Dict, Any

from .denoiser_proto import Denoiser

class VDenoiser(DiscreteVDDPMDenoiser, Denoiser):
  inner_model: UNet2DConditionModel
  sampling_dtype: torch.dtype
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: Tensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_v(
    self,
    sample: FloatTensor,
    timestep: Union[Tensor, float, int],
    encoder_hidden_states: FloatTensor,
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
    added_cond_kwargs: Dict[str, Any] = {},
    ) -> FloatTensor:
    out: UNet2DConditionOutput = self.inner_model(
      sample.to(self.inner_model.dtype),
      timestep.to(self.inner_model.dtype),
      encoder_hidden_states=encoder_hidden_states.to(self.inner_model.dtype),
      return_dict=return_dict,
      encoder_attention_mask=cross_attention_mask,
      added_cond_kwargs=added_cond_kwargs,
    )
    return out.sample.to(self.sampling_dtype)

  def sigma_to_t(self, sigma: FloatTensor, quantize=None) -> FloatTensor:
    return super().sigma_to_t(sigma, quantize=quantize).to(dtype=self.inner_model.dtype)

