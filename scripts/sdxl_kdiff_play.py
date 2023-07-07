from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch
from torch import FloatTensor, tensor, inference_mode
from typing import List
from logging import getLogger, Logger
from src.denoisers.v_denoiser import VDenoiser
from src.device import DeviceType, get_device_type
from src.schedule_params import get_alphas, get_alphas_cumprod, get_betas

logger: Logger = getLogger(__file__)

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

# https://birchlabs.co.uk/machine-learning#denoise-in-fp16-sample-in-fp32
sampling_dtype = torch.float32

unets: List[UNet2DConditionModel] = [UNet2DConditionModel.from_pretrained(
  f'stabilityai/stable-diffusion-xl-{expert}-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='unet',
).eval() for expert in ('base')]
# base_unet.to(device).eval()
base_unet, *_ = unets

tokenizers: List[CLIPTokenizer] = [CLIPTokenizer.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  subfolder=subfolder,
) for subfolder in ('tokenizer', 'tokenizer_2')]
tok_vit_l, tok_vit_big_g = tokenizers

# base uses ViT-L **and** ViT-bigG
# refiner uses just ViT-bigG
text_encoders: List[CLIPTextModel] = [CLIPTextModel.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder=subfolder,
).to(device).eval() for subfolder in ('text_encoder', 'text_encoder_2')]
vit_l, vit_big_g = text_encoders

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='vae',
)
vae_scale_factor: int = 2 ** (len(vae.config.block_out_channels) - 1)
# vae_scale_factor=8

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

tokenizeds: List[BatchEncoding] = [tokenizer(
  prompts,
  padding="max_length",
  max_length=tokenizer.model_max_length,
  truncation=True,
  return_overflowing_tokens=True,
) for tokenizer in tokenizers]
enc_vit_l, enc_vit_big_g = tokenizeds

wants_pooleds=(False, True)
embeddings: List[FloatTensor] = []
for tokenizer_ix, (tokenized, tokenizer, text_encoder, wants_pooled) in enumerate(zip(tokenizeds, tokenizers, text_encoders, wants_pooleds)):
  overflows: List[List[int]] = tokenized.data['overflowing_tokens']
  overflows_decoded: List[str] = tokenizer.batch_decode(overflows)
  num_truncateds: List[int] = tokenized.data['num_truncated_tokens']
  for prompt_ix, (overflow_decoded, num_truncated) in enumerate(zip(overflows_decoded, num_truncateds)):
    if num_truncated > 0:
      logger.warning(f"Prompt {prompt_ix} will be truncated, due to exceeding tokenizer {tokenizer_ix}'s length limit by {num_truncated} tokens. Overflowing portion of text was: <{overflow_decoded}>")
  with inference_mode():
    encoder_out: BaseModelOutputWithPooling = text_encoder.forward(
      input_ids=tensor(tokenized.input_ids, device=device),
      attention_mask=tensor(tokenized.attention_mask, device=device),
      output_hidden_states=not wants_pooled,
      return_dict=True,
    )
  embedding: FloatTensor = encoder_out.last_hidden_state if wants_pooled else encoder_out.hidden_states[-2],
  embeddings.append(embedding)
emb_vit_l, emb_vit_big_g = embeddings

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = VDenoiser(base_unet, alphas_cumprod, sampling_dtype)