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
    [dim for tup in (original_size, crop_coords_top_left, target_size) for dim in tup]
  ], device=device, dtype=dtype)

def get_time_ids_aesthetic(
  original_size: Dimensions,
  crop_coords_top_left: Dimensions,
  aesthetic_score: float,
  dtype: torch.dtype,
  device: DeviceType = torch.device('cpu'),
) -> FloatTensor:
  return tensor([
    [dim for tup in (original_size, crop_coords_top_left, (aesthetic_score,)) for dim in tup]
  ], device=device, dtype=dtype)