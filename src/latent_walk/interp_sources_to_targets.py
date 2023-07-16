from torch import FloatTensor, BoolTensor, tensor, lerp
from typing import List
from ..interpolation.slerp import slerp

def interp_sources_to_targets(
  sources: FloatTensor,
  targets: FloatTensor,
  quotients: List[float],
  wants_lerp: List[bool]
) -> FloatTensor:
  quotients_t: FloatTensor = tensor(quotients, dtype=sources.dtype).unsqueeze(-1).repeat(2, 1)
  wants_lerp_t: BoolTensor = tensor(wants_lerp).unsqueeze(-1).repeat(2, 1)

  slerped: FloatTensor = slerp(
    sources,
    targets,
    quotients_t,
  )
  lerped: FloatTensor = lerp(
    sources,
    targets,
    quotients_t,
  )
  interped: FloatTensor = lerped.where(wants_lerp_t, slerped)
  return interped