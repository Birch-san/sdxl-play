import torch
from torch import FloatTensor, tensor
from .device import DeviceType
from .dimensions import Dimensions

def get_time_ids(
  original_size: Dimensions,
  crop_coords_top_left: Dimensions,
  target_size: Dimensions,
  dtype: torch.dtype,
  device: DeviceType = torch.device('cpu'),
) -> FloatTensor:
  return tensor([
    [list(tup) for tup in (
      original_size,
      crop_coords_top_left,
      target_size,
    )]
  ], device=device, dtype=dtype)