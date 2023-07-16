import torch
from torch import LongTensor, stack, tensor
from typing import List

unique_num = 1 # reserve all-zeros embed for uncond
def mock_embed(prompts: List[str]) -> LongTensor:
  global unique_num
  start_ix=unique_num
  unique_num=unique_num+len(prompts)
  return stack([
    # using float32 because this script runs on-CPU, and CPU doesn't implement a float16 lerp.
    # in reality we'll run text encoders in float16, so we'll get float16 embeds.
    # *however* we might want to cast-to-fp32 to do the slerp.
    tensor((fill_value, 0, 1), dtype=torch.float32) for fill_value in range(
      start_ix,
      unique_num,
    )
  ])