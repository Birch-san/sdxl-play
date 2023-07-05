from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, BatchEncoding
import torch
from typing import Tuple, List
from logging import getLogger, Logger

logger: Logger = getLogger(__file__)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')

base_unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='unet',
)
base_unet.to(device).eval()

tokenizers: Tuple[CLIPTokenizer, CLIPTokenizer] = tuple(CLIPTokenizer.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  subfolder=subfolder,
) for subfolder in ('tokenizer', 'tokenizer_2'))
tok_vit_l, tok_vit_big_g = tokenizers

# base uses ViT-L **and** ViT-bigG
# refiner uses just ViT-bigG
text_encoders: Tuple[CLIPTextModel, CLIPTextModel] = tuple(CLIPTextModel.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder=subfolder,
).to(device).eval() for subfolder in ('text_encoder', 'text_encoder_2'))
vit_l, vit_big_g = text_encoders

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='vae',
)
vae_scale_factor: int = 2 ** (len(vae.config.block_out_channels) - 1) # 8

def round_down(num: int, divisor: int) -> int:
  return num - (num % divisor)

area = 1024**2
# SDXL's aspect-ratio bucketing trained on multiple-of-64 side lengths with close-to-1024**2 area
width_px: int = round_down(1024, 64)
height_px: int = round_down(area//width_px, 64)
width_lt: int = width_px // vae_scale_factor
height_lt: int = height_px // vae_scale_factor

batch_size = 1
cfg_scale = 5.

uncond_prompt: str = ''
prompt: str = 'astronaut meditating under waterfall, in swimming shorts'
prompts: List[str] = [uncond_prompt, prompt]

tokenizeds: Tuple[BatchEncoding, BatchEncoding] = tuple(tokenizer.__call__(
  prompts,
  padding="max_length",
  max_length=tokenizer.model_max_length,
  truncation=True,
  # return_tensors="pt",
  return_overflowing_tokens=True,
) for tokenizer in tokenizers)
enc_vit_l, enc_vit_big_g = tokenizeds

for tokenizer_ix, (encoded, tokenizer) in enumerate(zip(tokenizeds, tokenizers)):
  overflows: List[List[int]] = encoded.data['overflowing_tokens']
  overflows_decoded: List[str] = tokenizer.batch_decode(overflows)
  num_truncateds: List[int] = encoded.data['num_truncated_tokens']
  for prompt_ix, (overflow_decoded, num_truncated) in enumerate(zip(overflows_decoded, num_truncateds)):
    if num_truncated > 0:
      logger.warning(f"Prompt {prompt_ix} will be truncated, due to exceeding tokenizer {tokenizer_ix}'s length limit by {num_truncated} tokens. Overflowing portion of text was: <{overflow_decoded}>")

# for tokenizer, text_encoder in zip((tok_vit_l, tok_vit_big_g), (vit_l, vit_big_g)):
