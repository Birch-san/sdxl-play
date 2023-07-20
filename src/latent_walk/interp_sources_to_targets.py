import torch
from torch import FloatTensor, BoolTensor, tensor, lerp
from typing import List, Optional
from ..interpolation.slerp import slerp

def interp_sources_to_targets(
  sources: FloatTensor,
  targets: FloatTensor,
  quotients: List[float],
  wants_lerp: List[bool]
) -> FloatTensor:
  assert wants_lerp, 'need to specify per-source and per-target whether they want lerping'
  quotients_t: FloatTensor = tensor(quotients, dtype=sources.dtype, device=sources.device).repeat(2).reshape(-1, *(1,)*(sources.ndim-1))
  wants_lerp_t: BoolTensor = tensor(wants_lerp, dtype=torch.bool, device=sources.device).repeat(2).reshape(-1, *(1,)*(sources.ndim-1))

  all_lerp: bool = all(wants_lerp)
  all_slerp: bool = not any(wants_lerp)
  mixed: bool = not all_lerp and not all_slerp

  slerped: Optional[FloatTensor] = None if all_lerp else slerp(
    sources,
    targets,
    quotients_t,
  )
  lerped: Optional[FloatTensor] = None if all_slerp else lerp(
    sources,
    targets,
    quotients_t,
  )

  if mixed:
    interped: FloatTensor = lerped.where(wants_lerp_t, slerped)
    return interped
  
  if slerped is None:
    return lerped
  return slerped