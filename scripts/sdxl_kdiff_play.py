from contextlib import nullcontext
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.vae import DecoderOutput
from transformers import CLIPTextModel, CLIPTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch
from torch import FloatTensor, Generator, tensor, inference_mode, randn
from torch.backends.cuda import sdp_kernel
from torchvision.utils import save_image
from typing import List
from logging import getLogger, Logger
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m, sample_dpmpp_2m_sde

from src.denoisers.v_denoiser import VDenoiser
from src.denoisers.cfg_denoiser import CFGDenoiser
from src.device import DeviceType, get_device_type
from src.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from src.schedule_params import get_alphas, get_alphas_cumprod, get_betas
from src.latents_shape import LatentsShape
from src.added_cond import CondKwargs
from src.dimensions import Dimensions
from src.time_ids import get_time_ids
from src.device_ctx import to_device

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
).eval() for expert in ['base',]]
base_unet, *_ = unets
# base_unet = torch.compile(base_unet, mode="reduce-overhead", fullgraph=True)

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
).eval() for subfolder in ('text_encoder', 'text_encoder_2')]
vit_l, vit_big_g = text_encoders

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  # decoder gets NaN result in float16.
  # strictly speaking we probably only need to upcast the self-attention in the mid-block,
  # rather than everything. but diffusers doesn't expose a way to do this.
  torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
  use_safetensors=True,
  variant='fp16',
  subfolder='vae',
)
# the VAE decoder has a 512-dim self-attention in its mid-block. flash attn isn't supported for such a high dim,
# (no GPU has enough SRAM to compute that), so it runs without memory-efficient attn. slicing prevents OOM.
# TODO: rather than slicing *everything* (submit smaller image):
#       would it be faster to just slice the *attention* (serialize into n groups of (batch, head))?
#       because attention is likely to be the main layer at threat of encountering OOM.
#       i.e. vae.set_attn_processor(SlicedAttnProcessor())
vae.enable_slicing()
# this script won't be using the encoder
del vae.encoder
vae_scale_factor: int = 1 << (len(vae.config.block_out_channels) - 1)
# vae_scale_factor=8

def round_down(num: int, divisor: int) -> int:
  return num - (num % divisor)

area = 1024**2
# SDXL's aspect-ratio bucketing trained on multiple-of-64 side lengths with close-to-1024**2 area
width_px: int = round_down(1024, 64)
height_px: int = round_down(area//width_px, 64)
width_lt: int = width_px // vae_scale_factor
height_lt: int = height_px // vae_scale_factor

# TODO: is it height, width or the other way around?
# note: I think these conditions were mainly useful at training time
# (to teach SDXL to distinguish a native-resolution image from an upscaled one).
# I think at inference time you just want "a native-resolution image".
# so I think setting both original and target to (height_px, width_px) makes sense.
# and crop(0,0) (i.e. a well-framed image) make sense.
# TODO: should non-square images condition on orig=targ=(height_px, width_px),
#       or should they use orig=targ=(1024, 1024)?
original_size = Dimensions(height_px, width_px)
target_size = Dimensions(height_px, width_px)
crop_coords_top_left = Dimensions(0, 0)

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
  with to_device(text_encoder, device), inference_mode(), sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext:
    encoder_out: BaseModelOutputWithPooling = text_encoder.forward(
      input_ids=tensor(tokenized.input_ids, device=device),
      attention_mask=tensor(tokenized.attention_mask, device=device),
      output_hidden_states=not wants_pooled,
      return_dict=True,
    )
  embedding: FloatTensor = encoder_out.last_hidden_state if wants_pooled else encoder_out.hidden_states[-2]
  embeddings.append(embedding)
emb_vit_l, emb_vit_big_g = embeddings

time_ids: FloatTensor = get_time_ids(
  original_size=original_size,
  crop_coords_top_left=crop_coords_top_left,
  target_size=target_size,
  dtype=emb_vit_l.dtype,
  device=device,
)

added_cond_kwargs = CondKwargs(
  text_embeds=emb_vit_big_g,
  time_ids=time_ids.expand(emb_vit_big_g.size(0), -1),
)

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = VDenoiser(base_unet, alphas_cumprod, sampling_dtype)

denoiser = CFGDenoiser(
  denoiser=unet_k_wrapped,
  cross_attention_conds=embedding,
  added_cond_kwargs=added_cond_kwargs,
)

schedule_template = KarrasScheduleTemplate.Mastering
schedule: KarrasScheduleParams = get_template_schedule(
  schedule_template,
  model_sigma_min=unet_k_wrapped.sigma_min,
  model_sigma_max=unet_k_wrapped.sigma_max,
  device=unet_k_wrapped.sigmas.device,
  dtype=unet_k_wrapped.sigmas.dtype,
)

steps, sigma_max, sigma_min, rho = schedule.steps, schedule.sigma_max, schedule.sigma_min, schedule.rho
sigmas: FloatTensor = get_sigmas_karras(
  n=steps,
  sigma_max=sigma_max.cpu(),
  sigma_min=sigma_min.cpu(),
  rho=rho,
  device=device,
).to(sampling_dtype)

# note: if you ever change this script into img2img, then you will want to start the
# denoising from a later sigma than sigma_max.
start_sigma = sigma_max

latents_shape = LatentsShape(base_unet.in_channels, height_lt, width_lt)

seed = 42
# we generate with CPU random so that results can be reproduced across platforms
generator = Generator(device='cpu')

latents = randn((1, latents_shape.channels, latents_shape.height, latents_shape.width), dtype=sampling_dtype, device='cpu', generator=generator).to(device)
latents *= start_sigma

noise_sampler = BrownianTreeNoiseSampler(
  latents,
  sigma_min=sigma_min,
  sigma_max=start_sigma,
  # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
  # I'm just re-using it because it's a convenient arbitrary number
  seed=seed,
)

with inference_mode(), to_device(base_unet, device), sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext:
  denoised_latents: FloatTensor = sample_dpmpp_2m(
    denoiser,
    latents,
    sigmas,
    # noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers such as sample_dpmpp_2m_sde
    # callback=callback,
  ).to(vae.dtype)

denoised_latents = denoised_latents / vae.config.scaling_factor

vae.to(device)
# cannot use flash attn because VAE decoder's self-attn has head_dim 512
with inference_mode():#, sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext:
  decoder_out: DecoderOutput = vae.decode(denoised_latents.to(vae.dtype))
sample: FloatTensor = decoder_out.sample
for ix, decoded in enumerate(sample):
  save_image(decoded, f'out/base_{ix}.png') 