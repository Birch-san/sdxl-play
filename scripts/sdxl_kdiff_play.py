from contextlib import nullcontext
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.vae import DecoderOutput
from transformers import CLIPPreTrainedModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
import torch
from torch import BoolTensor, FloatTensor, Generator, inference_mode, cat, randn, tensor, zeros
from torch.nn.functional import pad
from torch.backends.cuda import sdp_kernel
from typing import List, Union, Optional, Callable
from logging import getLogger, Logger
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m, sample_dpmpp_2m_sde
from os import makedirs, listdir
from os.path import join
import fnmatch
from pathlib import Path
from PIL import Image

from src.denoisers.eps_denoiser import EPSDenoiser
from src.denoisers.cfg_denoiser import CFGDenoiser
from src.denoisers.nocfg_denoiser import NoCFGDenoiser
from src.device import DeviceType, get_device_type
from src.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from src.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from src.latents_shape import LatentsShape
from src.added_cond import CondKwargs
from src.dimensions import Dimensions
from src.time_ids import get_time_ids
from src.device_ctx import to_device
from src.rgb_to_pil import rgb_to_pil

logger: Logger = getLogger(__file__)

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

# https://birchlabs.co.uk/machine-learning#denoise-in-fp16-sample-in-fp32
sampling_dtype = torch.float32
# sampling_dtype = torch.float16

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
vit_l: CLIPTextModel = CLIPTextModel.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='text_encoder',
).eval()

vit_big_g: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-0.9',
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='text_encoder_2',
).eval()

text_encoders: List[CLIPPreTrainedModel] = [vit_l, vit_big_g]

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
# https://arxiv.org/abs/2305.08891
# Common Diffusion Noise Schedules and Sample Steps are Flawed
# 3.4. Rescale Classifier-Free Guidance
cfg_rescale = 0.

force_zeros_for_empty_prompt = True
uncond_prompt: Optional[str] = None if force_zeros_for_empty_prompt else ''

negative_prompt: Optional[str] = uncond_prompt
# prompt: str = 'astronaut meditating under waterfall, in swimming shorts'
prompt: str = '90s anime sketch, girl wearing serafuku walking home, masterpiece, dramatic, wind'
prompts: List[str] = [
  *([] if negative_prompt is None else [negative_prompt]),
  prompt,
]

tokenizeds: List[BatchEncoding] = [tokenizer(
  prompts,
  padding="max_length",
  max_length=tokenizer.model_max_length,
  truncation=True,
  return_overflowing_tokens=True,
) for tokenizer in tokenizers]
enc_vit_l, enc_vit_big_g = tokenizeds

# we expect to get one embedding (of penultimate hidden states) per text encoder
embeddings: List[FloatTensor] = []
embedding_masks: List[BoolTensor] = []
# we expect to get just one pooled embedding, from vit_big_g
pooled_embed: Optional[FloatTensor] = None
for tokenizer_ix, (tokenized, tokenizer, text_encoder) in enumerate(zip(tokenizeds, tokenizers, text_encoders)):
  overflows: List[List[int]] = tokenized.data['overflowing_tokens']
  overflows_decoded: List[str] = tokenizer.batch_decode(overflows)
  num_truncateds: List[int] = tokenized.data['num_truncated_tokens']
  for prompt_ix, (overflow_decoded, num_truncated) in enumerate(zip(overflows_decoded, num_truncateds)):
    if num_truncated > 0:
      logger.warning(f"Prompt {prompt_ix} will be truncated, due to exceeding tokenizer {tokenizer_ix}'s length limit by {num_truncated} tokens. Overflowing portion of text was: <{overflow_decoded}>")

  input_ids=tensor(tokenized.input_ids, device=device)
  attention_mask=tensor(tokenized.attention_mask, device=device, dtype=torch.bool)
  embedding_masks.append(attention_mask)

  with to_device(text_encoder, device), inference_mode(), sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext:
    assert isinstance(text_encoder, CLIPTextModel) or isinstance(text_encoder, CLIPTextModelWithProjection), f'text_encoder of type "{type(text_encoder)}" was unexpectedly neither a CLIPTextModel nor a CLIPTextModelWithProjection'
    encoder_out: Union[BaseModelOutputWithPooling, CLIPTextModelOutput] = text_encoder(
      input_ids=input_ids,
      # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
      # attention_mask=attention_mask,
      output_hidden_states=True,
      return_dict=True,
    )
    embedding: FloatTensor = encoder_out.hidden_states[-2]
    if isinstance(encoder_out, CLIPTextModelOutput): # returned by CLIPTextModelWithProjection, e.g. vit_big_g
      assert pooled_embed is None, 'we only expected one of the text encoders (vit_big_g) to be a CLIPTextModelWithProjection capable of providing a pooled embedding.'
      pooled_embed: FloatTensor = encoder_out.text_embeds
  embeddings.append(embedding)

assert pooled_embed is not None
concat_embed: FloatTensor = cat(embeddings, dim=-1)
embedding_mask: BoolTensor = cat(embedding_masks, dim=-1)

if negative_prompt is None and cfg_scale > 1.:
  # uncond_embed: FloatTensor = zeros((1, *concat_embed.shape[1:]), dtype=concat_embed.dtype, device=device)
  # concat_embed = cat([uncond_embed, concat_embed], dim=0)
  concat_embed = pad(concat_embed, pad=(0,0, 0,0, 1,0), mode='constant')

  # uncond_pooled: FloatTensor = zeros((1, *pooled_embed.shape[1:]), dtype=pooled_embed.dtype, device=device)
  # pooled_embed = cat([uncond_pooled, pooled_embed], dim=0)
  pooled_embed = pad(pooled_embed, pad=(0,0, 1,0), mode='constant')

  embedding_mask = pad(pooled_embed, pad=(0,0, 1,0), mode='constant', value=True)

time_ids: FloatTensor = get_time_ids(
  original_size=original_size,
  crop_coords_top_left=crop_coords_top_left,
  target_size=target_size,
  dtype=concat_embed.dtype,
  device=device,
)

added_cond_kwargs = CondKwargs(
  text_embeds=pooled_embed,
  time_ids=time_ids.expand(concat_embed.size(0), -1),
)

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = EPSDenoiser(base_unet, alphas_cumprod, sampling_dtype)

if cfg_scale > 1:
  denoiser = CFGDenoiser(
    denoiser=unet_k_wrapped,
    cross_attention_conds=concat_embed,
    added_cond_kwargs=added_cond_kwargs,
    cfg_scale=cfg_scale,
    guidance_rescale=cfg_rescale,
    # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
    # cross_attention_mask=embedding_mask,
  )
else:
  denoiser = NoCFGDenoiser(
    denoiser=unet_k_wrapped,
    cross_attention_conds=concat_embed,
    added_cond_kwargs=added_cond_kwargs,
    # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
    # cross_attention_mask=embedding_mask,
  )

schedule_template = KarrasScheduleTemplate.Science
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

# here's how to use non-Karras sigmas
# steps=25
# sigmas = unet_k_wrapped.get_sigmas(steps)
# sigma_max, sigma_min = sigmas[0], sigmas[-2]

print(f"sigmas (unquantized):\n{', '.join(['%.4f' % s.item() for s in sigmas])}")
sigmas_quantized = torch.cat([
  quantize_to(sigmas[:-1], unet_k_wrapped.sigmas),
  zeros((1), device=sigmas.device, dtype=sigmas.dtype)
])
print(f"sigmas (quantized):\n{', '.join(['%.4f' % s.item() for s in sigmas_quantized])}")
# TODO: discretize sigmas?
# sigmas = sigmas_quantized

# note: if you ever change this script into img2img, then you will want to start the
# denoising from a later sigma than sigma_max.
start_sigma = sigmas[0]
# start_sigma = sigma_max
# due to float16 precision, the sigma_max we pass into our schedule, may get modified
# start_sigma = schedule.sigma_max
# diffusers EulerDiscreteScheduler#init_noise_sigma does this for some reason:
# start_sigma = (sigma_max ** 2 + 1) ** 0.5

latents_shape = LatentsShape(base_unet.in_channels, height_lt, width_lt)

seed = 42
# we generate with CPU random so that results can be reproduced across platforms
generator = Generator(device='cpu').manual_seed(seed)

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
  # TODO: guidance_rescale
  denoised_latents: FloatTensor = sample_dpmpp_2m(
    denoiser,
    latents,
    sigmas,
    # noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers such as sample_dpmpp_2m_sde
    # callback=callback,
  ).to(vae.dtype)

denoised_latents = denoised_latents / vae.config.scaling_factor

# we avoid using our to_device() context manager, because we have no further large tasks;
# don't want to pay to transfer VAE back to CPU upon context exit
vae.to(device)
# cannot use flash attn because VAE decoder's self-attn has head_dim 512
with inference_mode():#, sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext:
  decoder_out: DecoderOutput = vae.decode(denoised_latents.to(vae.dtype))

out_dir = 'out'
makedirs(out_dir, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0
out_stem: str = f'{next_ix:05d}_base_{prompt.split(",")[0]}_{seed}'

sample: FloatTensor = decoder_out.sample.div(2).add(.5).clamp(0,1)
for ix, decoded in enumerate(sample):
  # if you want lossless images: consider png
  out_name: str = join(out_dir, f'{out_stem}.jpg')
  img: Image = rgb_to_pil(decoded)
  img.save(out_name, subsampling=0, quality=95)
  print(f'Saved image: {out_name}')