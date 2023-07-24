from diffusers.models.attention import Attention
from torch.nn import Module, Linear
from torch import cat

from .flash_attn_processor import FlashAttnProcessor
from .flash_attn_qkv_packed_processor import FlashAttnQKVPackedProcessor

def apply_flash_attn_processor(mod_: Module) -> None:
  cross_attn_processor = FlashAttnProcessor()
  self_attn_processor = FlashAttnQKVPackedProcessor()
  def set_flash_attn_processor(mod: Module) -> None:
    if isinstance(mod, Attention):
      # relying on a side-channel to determine (unreliably) whether a layer is self-attention
      if mod.to_k.in_features == mod.to_q.in_features:
        # probably self-attention
        mod.to_qkv = Linear(mod.to_q.in_features, mod.to_q.out_features*3, dtype=mod.to_q.weight.dtype, device=mod.to_q.weight.data.device)
        mod.to_qkv.weight.data = cat([mod.to_q.weight, mod.to_k.weight, mod.to_v.weight]).detach()
        del mod.to_q, mod.to_k, mod.to_v
        mod.set_processor(self_attn_processor)
      else:
        mod.set_processor(cross_attn_processor)
  mod_.apply(set_flash_attn_processor)