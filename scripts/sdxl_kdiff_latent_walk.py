from contextlib import nullcontext
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.vae import DecoderOutput
from diffusers.models.attention import Attention
from easing_functions import CubicEaseInOut
from transformers import CLIPPreTrainedModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import torch
from torch import BoolTensor, FloatTensor, LongTensor, Generator, inference_mode, cat, randn, tensor, zeros, ones, stack
from torch.nn.functional import pad
from torch.backends.cuda import sdp_kernel
from torch.nn import Module, Linear
from typing import List, Optional, Callable, Set, Dict, Any
from logging import getLogger, Logger
from k_diffusion.sampling import get_sigmas_karras, sample_dpmpp_2m#, sample_dpmpp_2m_sde, sample_euler, BrownianTreeNoiseSampler
from os import makedirs, listdir
from os.path import join
import fnmatch
from pathlib import Path
from PIL import Image
from functools import partial
from time import perf_counter

from src.iteration.batched import batched
from src.denoisers.denoiser_proto import Denoiser
from src.denoisers.denoiser_factory import DenoiserFactory, DenoiserFactoryFactory
from src.denoisers.dispatch_denoiser import DispatchDenoiser, IdentifiedDenoiser
from src.denoisers.eps_denoiser import EPSDenoiser
from src.denoisers.cfg_denoiser import CFGDenoiser
from src.denoisers.nocfg_denoiser import NoCFGDenoiser
from src.device import DeviceType, get_device_type
from src.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
# from src.diffusers_schedules import get_diffusers_euler_discrete_schedule
from src.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to#, get_sigmas
from src.latents_shape import LatentsShape
from src.added_cond import CondKwargs
from src.dimensions import Dimensions
from src.time_ids import get_time_ids, get_time_ids_aesthetic
from src.device_ctx import to_device
from src.rgb_to_pil import rgb_to_pil
from src.embed_mgmt.tokenize import tokenize, TokenizerOutput
from src.embed_mgmt.embed import embed, EmbedderOutput
from src.interpolation.intersperse_linspace import intersperse_linspace, InterpSpec
from src.interpolation.in_between import ManneredInBetweenParams
from src.interpolation.inter_prompt import InterPrompt
from src.interpolation.interp_strategy import InterpStrategy
from src.interpolation.interp_manner import InterpManner, QuotientModifier
from src.iteration.batched import batched
from src.sample_spec.prompts import CFGPrompts, NoCFGPrompts, PromptType
from src.sample_spec.sample_spec import SampleSpec
from src.embed_mgmt.make_get_embed import make_get_embed
from src.embed_mgmt.get_embed import GetEmbed, EmbeddingFromOptionalText
from src.embed_mgmt.get_prompt_text import CFGPole, InterpPole, get_prompt_text
from src.embed_mgmt.embed_batch import embed_batch
from src.embed_mgmt.embed_cache import EmbedCache
from src.latent_walk.interp_sources_to_targets import interp_sources_to_targets
from src.clip_pooling import forward_penultimate_hidden_state, pool_and_project_last_hidden_state
from src.attn.apply_flash_attn_processor import apply_flash_attn_processor
from src.attn.flash_attn_processor import FlashAttnProcessor

logger: Logger = getLogger(__file__)

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

# https://birchlabs.co.uk/machine-learning#denoise-in-fp16-sample-in-fp32
sampling_dtype = torch.float32
# sampling_dtype = torch.float16

# if you have low VRAM, then we can swap Unets and VAE into VRAM and back as needed.
# if you have high VRAM: don't bother with this because swapping costs time.
# if you're on a Mac: don't bother with this; VRAM and RAM are the same thing.
swap_models = False

use_wdxl = False

stability_model_name = 'stabilityai/stable-diffusion-xl-base-0.9'
wdxl_model_name = 'Birchlabs/waifu-diffusion-xl-unofficial'
default_model_name = stability_model_name
base_unet_model_name = wdxl_model_name if use_wdxl else stability_model_name

use_refiner = True
unets: List[UNet2DConditionModel] = [UNet2DConditionModel.from_pretrained(
  unet_name,
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='unet',
).eval() for unet_name in [
  base_unet_model_name,
  *(['stabilityai/stable-diffusion-xl-refiner-0.9'] * use_refiner),
  ]
]
base_unet: UNet2DConditionModel = unets[0]
refiner_unet: Optional[UNet2DConditionModel] = unets[1] if use_refiner else None

use_xformers_attn = False
if use_xformers_attn:
  from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
  for unet in unets:
    unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

use_flash_attn = False
if use_flash_attn:
  for unet in unets:
    processor = FlashAttnProcessor()
    unet.set_attn_processor(processor)

use_flash_attn_qkv_packed = True
if use_flash_attn_qkv_packed:
  for unet in unets:
    apply_flash_attn_processor(unet)

measure_sampling = False

compile = False
if compile:
  for unet in unets:
    torch.compile(unet, mode='reduce-overhead', fullgraph=True)

tokenizers: List[CLIPTokenizer] = [CLIPTokenizer.from_pretrained(
  default_model_name,
  subfolder=subfolder,
) for subfolder in ('tokenizer', 'tokenizer_2')]
tok_vit_l, tok_vit_big_g = tokenizers

# base uses ViT-L **and** ViT-bigG
# refiner uses just ViT-bigG
vit_l: CLIPTextModel = CLIPTextModel.from_pretrained(
  default_model_name,
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='text_encoder',
).eval()

vit_big_g: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
  default_model_name,
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16',
  subfolder='text_encoder_2',
).eval()

text_encoders: List[CLIPPreTrainedModel] = [vit_l, vit_big_g]

use_ollin_vae = True
vae_kwargs: Dict[str, Any] = {
  'torch_dtype': torch.float16,
} if use_ollin_vae else {
  'variant': 'fp16',
  'subfolder': 'vae',
  # decoder gets NaN result in float16.
  'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
}

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  'madebyollin/sdxl-vae-fp16-fix' if use_ollin_vae else default_model_name,
  use_safetensors=True,
  **vae_kwargs,
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
original_size = Dimensions(height_px*4, width_px*4) if use_wdxl else Dimensions(height_px, width_px)
target_size = Dimensions(height_px, width_px)
crop_coords_top_left = Dimensions(0, 0)
aesthetic_score = 6.
negative_aesthetic_score = 2.5

cfg_scale = 12. if use_wdxl else 5.
refiner_cfg_scale = 5. if use_wdxl else cfg_scale
# https://arxiv.org/abs/2305.08891
# Common Diffusion Noise Schedules and Sample Steps are Flawed
# 3.4. Rescale Classifier-Free Guidance
cfg_rescale = 0.
# cfg_rescale = 0.7

# apparently zero-uncond was just an inference experiment; it's not actually how SDXL was trained.
# https://twitter.com/s_alt_acc/status/1683627077315227648
# interestingly, diffusers default is to enable this for base UNet, disable for refiner.
# it'd be a little fiddly in this script to make base and refiner separately configurable.
# fortunately it sounds like the simpler way (use empty string, for both UNets) is the correct way.
force_zeros_for_empty_prompt = False

if use_wdxl:
  uncond_prompt: str = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name'
else:
  uncond_prompt: Optional[str] = None if force_zeros_for_empty_prompt else ''

keyframes: List[PromptType] = [
  CFGPrompts(
    uncond_prompt=uncond_prompt,
    prompt=prompt,
  ) if cfg_scale > 1 else NoCFGPrompts(
    prompt=prompt,
  ) for prompt in [
    'girl with dragon, flying over water, masterpiece, ghibli, reflection, grinning',
    'the dragon attacks at night, masterpiece, dramatic, highly detailed, high dynamic range',
    'the dragon attacks Neo-Tokyo at night, masterpiece, dramatic, highly detailed, high dynamic range',
    'watercolour illustration of Japanese lanterns floating down river, festival, moonlight',
    'thousands of fireflies on the river at night, moonlight, grass, trees, 4k, dslr, cinematic, masterpiece',
    'photo of astronaut meditating under waterfall, in swimming shorts, moonlight, breathtaking, 4k, dslr, cinematic, global illumination, realistic, highly detailed',
    'character portrait of steampunk lady, ponytail, hat, masterpiece, intricate detail',
    'art of refined steampunk gentleman, wearing suspenders, holding timepiece, monocle, well-groomed beard and moustache',
    'an explorer struggles to get out of quicksand',
    'an explorer struggles to get out of quicksand, desert',
    'illustration of cavemen habitat, cooking pots, sunset, long shadows, wide angle',
    'photograph of a torii gate in the rain, high up a mountain pass in Kyoto',
    'photograph of a torii gate in the rain, high up a mountain pass in Kyoto, vaporwave',
    'fish swimming up waterfall of mercury, aurora borealis',
    'illustration of gamer girl with blue hair, sitting in chair with knees up, focusing intently on computer',
    'illustration of gamer girl with pink hair, sitting in chair with knees up, focusing intently on computer',
    'girl with dragon, flying over water, masterpiece, ghibli, reflection, grinning',
  ]
]

# quotient_modifiers: List[QuotientModifier] = [lambda x:x, CubicEaseInOut()]
modifier: QuotientModifier = lambda x:x

interp_specs: List[InterpSpec[InterpManner]] = [InterpSpec[InterpManner](
  steps=100,
  manner=InterpManner(
    quotient_modifier=modifier,
    # SDXL is conditioned on penultimate hidden states, which haven't been normalized.
    # this means it doesn't make sense to slerp (as fun as that would be).
    strategy=InterpStrategy.Lerp,
  ),
) for _ in range(len(keyframes)-1)]
# ) for _, modifier in zip(range(len(keyframes)-1), quotient_modifiers)]

def make_inbetween(params: ManneredInBetweenParams[PromptType, InterpManner]) -> InterPrompt[PromptType]:
  # apply easing function
  modified_quotient: float = params.manner.quotient_modifier(params.quotient)
  return InterPrompt[PromptType](
    from_=params.from_,
    to=params.to,
    quotient=modified_quotient,
    strategy=params.manner.strategy,
  )

frames: List[SampleSpec] = intersperse_linspace(
  keyframes=keyframes,
  make_inbetween=make_inbetween,
  interp_specs=interp_specs,
)

all_zeros_hidden_states: List[Optional[FloatTensor]] = [
  zeros(
    (text_encoder.config.max_position_embeddings, text_encoder.config.hidden_size),
    dtype=text_encoder.dtype,
    device=device,
  )
for text_encoder in text_encoders] if force_zeros_for_empty_prompt else [None, None]

all_ones_masks: List[Optional[FloatTensor]] = [
  ones(
    (text_encoder.config.max_position_embeddings),
    dtype=torch.bool,
    device=device,
  )
for text_encoder in text_encoders] if force_zeros_for_empty_prompt else [None, None]

vit_big_g_all_zeros_pool: FloatTensor = zeros(
  (vit_big_g.config.hidden_size,),
  dtype=vit_big_g.dtype,
  device=device,
)

# all_zeros_pooled_clip_vit_big_g

base_time_ids: FloatTensor = get_time_ids(
  original_size=original_size,
  crop_coords_top_left=crop_coords_top_left,
  target_size=target_size,
  dtype=base_unet.dtype,
  device=device,
)
refiner_time_ids: FloatTensor = cat([
  get_time_ids_aesthetic(
    original_size=original_size,
    crop_coords_top_left=crop_coords_top_left,
    aesthetic_score=score,
    dtype=base_unet.dtype,
    device='cpu',
  ) for score in [*([negative_aesthetic_score] if cfg_scale > 1. else []), aesthetic_score]
]).to(device)

alphas_cumprod: FloatTensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)

unets_k_wrapped: List[EPSDenoiser] = [
  EPSDenoiser(
    unet,
    alphas_cumprod,
    sampling_dtype,
  ) for unet in [base_unet, *([refiner_unet] * use_refiner)]
]
base_unet_k_wrapped: EPSDenoiser = unets_k_wrapped[0]
refiner_unet_k_wrapped: Optional[EPSDenoiser] = unets_k_wrapped[1] if use_refiner else None

if cfg_scale > 1:
  def make_cfg_denoiser(
    delegate: Denoiser,
    cross_attention_conds: FloatTensor,
    added_cond_kwargs: CondKwargs,
    cfg_scale: float,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> CFGDenoiser:
    return CFGDenoiser(
      denoiser=delegate,
      cross_attention_conds=cross_attention_conds,
      added_cond_kwargs=added_cond_kwargs,
      cfg_scale=cfg_scale,
      guidance_rescale=cfg_rescale,
      cross_attention_mask=cross_attention_mask,
    )
  denoiser_factory_factory: DenoiserFactoryFactory[CFGDenoiser] = lambda delegate: partial(make_cfg_denoiser, delegate)
else:
  def make_no_cfg_denoiser(
    delegate: Denoiser,
    cross_attention_conds: FloatTensor,
    added_cond_kwargs: CondKwargs,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> CFGDenoiser:
    return NoCFGDenoiser(
      denoiser=delegate,
      cross_attention_conds=cross_attention_conds,
      added_cond_kwargs=added_cond_kwargs,
      cross_attention_mask=cross_attention_mask,
    )
  denoiser_factory_factory: DenoiserFactoryFactory[NoCFGDenoiser] = lambda delegate: partial(make_no_cfg_denoiser, delegate)

base_denoiser_factory: DenoiserFactory[Denoiser] = denoiser_factory_factory(base_unet_k_wrapped)
refiner_denoiser_factory: Optional[DenoiserFactory[Denoiser]] = denoiser_factory_factory(refiner_unet_k_wrapped) if use_refiner else None

schedule_template = KarrasScheduleTemplate.CudaMasteringMaximizeRefiner
# schedule_template = KarrasScheduleTemplate.Mastering
schedule: KarrasScheduleParams = get_template_schedule(
  schedule_template,
  model_sigma_min=base_unet_k_wrapped.sigma_min,
  model_sigma_max=base_unet_k_wrapped.sigma_max,
  device=base_unet_k_wrapped.sigmas.device,
  dtype=base_unet_k_wrapped.sigmas.dtype,
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
# sigmas = base_unet_k_wrapped.get_sigmas(steps)
# sigma_max, sigma_min = sigmas[0], sigmas[-2]

print(f"sigmas (unquantized):\n{', '.join(['%.4f' % s.item() for s in sigmas])}")
sigmas_quantized = pad(quantize_to(sigmas[:-1], base_unet_k_wrapped.sigmas), pad=(0, 1))
print(f"sigmas (quantized):\n{', '.join(['%.4f' % s.item() for s in sigmas_quantized])}")
sigmas = sigmas_quantized

if use_refiner:
  # (SDXL technical report, 2.5)
  # refiner specializes in the final 200 timesteps of the denoising schedule,
  # i.e. sigmas starting from:
  #   0.5692849159240723
  # I assume this corresponds to img2img strength of 0.2
  refine_from_sigma: float = base_unet_k_wrapped.sigmas[199].item()
  print(f"the highest sigma within the refiner's training was training distribution is: {refine_from_sigma:.4f} ")

  base_purview: BoolTensor = sigmas>refine_from_sigma
  refiner_purview: BoolTensor = sigmas<=refine_from_sigma
  base_duty: int = base_purview.sum()
  refiner_duty: int = refiner_purview[:-1].sum() # zero doesn't count
  print(f"base model will be active during {base_duty} sigmas:\n{', '.join(['%.4f' % s.item() for s in sigmas[base_purview]])}")
  print(f"refiner model will be active during {refiner_duty} sigmas:\n{', '.join(['%.4f' % s.item() for s in sigmas[refiner_purview][:-1]])}")

# note: if you ever change this script into img2img, then you will want to start the
# denoising from a later sigma than sigma_max.
start_sigma = sigmas[0]

out_dir = 'out'
makedirs(out_dir, exist_ok=True)

out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0

latents_shape = LatentsShape(base_unet.config.in_channels, height_lt, width_lt)

# we generate with CPU random so that results can be reproduced across platforms
generator = Generator(device='cpu')

max_batch_size = 8

seed = 14600

if not swap_models:
  base_unet.to(device)
  # # TODO: uncomment VAE bits once we're done with text encoding and Unet
  vae.to(device)
  if use_refiner:
    refiner_unet.to(device)
  for text_encoder in text_encoders:
    text_encoder.to(device)

img_provenance: str = 'refined' if use_refiner else 'base'

print(f'Generating {len(frames)} images, in batches of {max_batch_size}')

embed_caches: List[EmbedCache[FloatTensor]] = [EmbedCache[FloatTensor]() for _ in range(2)]
mask_caches: List[EmbedCache[BoolTensor]] = [EmbedCache[BoolTensor]() for _ in range(2)]
vit_big_g_input_ids_cache = EmbedCache[LongTensor]()
vit_big_g_pool_cache = EmbedCache[FloatTensor]()
vit_l_emb_cache, vit_big_g_emb_cache = embed_caches
vit_l_mask_cache, vit_big_g_mask_cache = mask_caches

for batch_ix, batch_frames in enumerate(batched(frames, max_batch_size)):
  batch_size: int = len(batch_frames)
  print(f'Generating a batch of {batch_size} images, prompts {[frame.from_.prompt.split(",")[0] if isinstance(frame, InterPrompt) else frame.prompt.split(",")[0] for frame in batch_frames]}, quotients {[f"{frame.quotient:.02f}" if isinstance(frame, InterPrompt) else "0" for frame in batch_frames]}')
  latents: FloatTensor = randn(
    (
      latents_shape.channels,
      latents_shape.height,
      latents_shape.width
    ),
    dtype=sampling_dtype,
    device=generator.device,
    generator=generator.manual_seed(seed),
  ).expand(batch_size, -1, -1, -1).to(device)
  latents *= start_sigma

  out_stems: List[str] = [
    f'{(next_ix + batch_ix*max_batch_size + sample_ix):05d}_{img_provenance}_{frame.from_.prompt.split(",")[0] if isinstance(frame, InterPrompt) else frame.prompt.split(",")[0]}_{f"{frame.quotient:.02f}_" if isinstance(frame, InterPrompt) else ""}{seed}'
    for sample_ix, frame in enumerate(batch_frames)
  ]

  new_prompt_texts: Set[str] = set()
  new_prompt_texts_ordered: List[str] = []
  retained_embed_ixs: Set[int] = set()

  for frame in batch_frames:
    sample_prompts: List[PromptType] = [frame.from_, frame.to] if isinstance(frame, InterPrompt) else [frame]
    for prompt in sample_prompts:
      for prompt_text in [*[prompt.uncond_prompt] * isinstance(prompt, CFGPrompts), prompt.prompt]:
        if prompt_text is None:
          continue
        cache_ix: Optional[int] = vit_l_emb_cache.get_cache_ix(prompt_text)
        if cache_ix is None:
          if prompt_text not in new_prompt_texts:
            new_prompt_texts.add(prompt_text)
            new_prompt_texts_ordered.append(prompt_text)
        else:
          retained_embed_ixs.add(cache_ix)
  
  new_vit_big_g_input_ids: Optional[LongTensor] = None
  new_vit_big_g_pools: Optional[FloatTensor] = None
  if new_prompt_texts_ordered:
    new_masks: List[BoolTensor] = []
    new_hidden_states: List[FloatTensor] = []
    for tokenizer_ix, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
      tokenizer_output: TokenizerOutput = tokenize(
        tokenizer=tokenizer,
        prompts=new_prompt_texts_ordered,
        device=device,
      )
      attention_mask: BoolTensor = tokenizer_output.attention_mask
      new_masks.append(attention_mask)
      if tokenizer is tok_vit_big_g:
        new_vit_big_g_input_ids = tokenizer_output.input_ids
      with to_device(text_encoder, device) if swap_models else nullcontext(), inference_mode(), sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext():
        embed_out: EmbedderOutput = embed(
          text_encoder=text_encoder,
          tokenizer_output=tokenizer_output,
        )
        new_hidden_states.append(embed_out.penultimate_hidden_states)
      if text_encoder is vit_big_g:
        new_vit_big_g_pools = embed_out.norm_pooled
    new_encoded_vit_l, new_encoded_vit_big_g = new_hidden_states
  else:
    new_masks: List[Optional[BoolTensor]] = [None, None]
    new_hidden_states: List[Optional[FloatTensor]] = [None, None]
  new_encoded_vit_l, new_encoded_vit_big_g = new_hidden_states

  if vit_l_emb_cache.cache is None:
    retained_hidden_states: List[Optional[FloatTensor]] = [None, None]
    retained_masks: List[Optional[BoolTensor]] = [None, None]
    retained_vit_big_g_input_ids: Optional[LongTensor] = None
    retained_vit_big_g_pools: Optional[FloatTensor] = None
  else:
    retained_hidden_states: List[FloatTensor] = [
      embed_cache.cache.index_select(0, tensor(list(retained_embed_ixs), dtype=torch.int64, device=device))
      for embed_cache in embed_caches
    ]
    retained_masks: List[BoolTensor] = [
      mask_cache.cache.index_select(0, tensor(list(retained_embed_ixs), dtype=torch.int64, device=device))
      for mask_cache in mask_caches
    ]
    retained_vit_big_g_input_ids: LongTensor = vit_big_g_input_ids_cache.cache.index_select(0, tensor(list(retained_embed_ixs), dtype=torch.int64, device=device))
    retained_vit_big_g_pools: FloatTensor = vit_big_g_pool_cache.cache.index_select(0, tensor(list(retained_embed_ixs), dtype=torch.int64, device=device))
  retained_vit_l_embeds, retained_vit_big_g_embeds = retained_hidden_states
  retained_vit_l_masks, retained_vit_big_g_masks = retained_masks
  
  interped_penult_hidden_states: List[FloatTensor] = []
  embedding_masks: List[BoolTensor] = []
  # these are just the input IDs of samples which require interping
  input_ids_: List[Optional[LongTensor]] = []
  pools: List[Optional[FloatTensor]] = []
  for (
    retained_hidden_states_,
    new_hidden_states_,
    retained_masks_,
    new_masks_,
    retained_input_ids,
    new_input_ids,
    retained_pools,
    new_pools,
    emb_cache,
    mask_cache,
    input_ids_cache,
    pool_cache,
    all_zeros_hidden_states_,
    all_ones_mask,
    all_zeros_pool_,
    ) in zip(
    retained_hidden_states,
    new_hidden_states,
    retained_masks,
    new_masks,
    [None, retained_vit_big_g_input_ids],
    [None, new_vit_big_g_input_ids],
    [None, retained_vit_big_g_pools],
    [None, new_vit_big_g_pools],
    embed_caches,
    mask_caches,
    [None, vit_big_g_input_ids_cache],
    [None, vit_big_g_pool_cache],
    all_zeros_hidden_states,
    all_ones_masks,
    [None, vit_big_g_all_zeros_pool],
  ):
    assert retained_hidden_states is not None or new_hidden_states is not None
    next_emb_cache: FloatTensor = cat([t for t in [retained_hidden_states_, new_hidden_states_] if t is not None])
    next_cache_prompts: List[str] = [prompt for ix, prompt in enumerate(emb_cache.prompts) if ix in retained_embed_ixs] + new_prompt_texts_ordered
    emb_cache.update_cache(
      cache=next_emb_cache,
      prompts=next_cache_prompts,
    )
    next_mask_cache: BoolTensor = cat([t for t in [retained_masks_, new_masks_] if t is not None])
    mask_cache.update_cache(
      cache=next_mask_cache,
      prompts=next_cache_prompts,
    )
    masks: BoolTensor = stack([
      all_ones_mask if prompt is None else mask_cache.get_by_prompt(prompt) for prompt in [
        *[get_prompt_text(frame, interp_pole='from', cfg_pole='uncond') for frame in batch_frames] * (cfg_scale > 1),
        *[get_prompt_text(frame, interp_pole='from', cfg_pole='cond') for frame in batch_frames],
      ]
    ])
    embedding_masks.append(masks)

    if input_ids_cache is None:
      input_ids_.append(None)
    else:
      next_input_ids_cache: LongTensor = cat([t for t in [retained_input_ids, new_input_ids] if t is not None])
      input_ids_cache.update_cache(
        cache=next_input_ids_cache,
        prompts=next_cache_prompts,
      )
      input_ids_for_interping: List[int] = [
        input_ids_cache.get_by_prompt(prompt) for prompt in [
          *[frame.from_.uncond_prompt for frame in batch_frames if isinstance(frame, InterPrompt) and frame.from_.uncond_prompt is not None] * (cfg_scale > 1),
          *[frame.from_.prompt for frame in batch_frames if isinstance(frame, InterPrompt)],
        ]
      ]
      input_ids_for_interping_t: Optional[LongTensor] = stack(input_ids_for_interping) if input_ids_for_interping else None
      input_ids_.append(input_ids_for_interping_t)
    
    if pool_cache is None:
      pools.append(None)
    else:
      next_pool_cache: FloatTensor = cat([t for t in [retained_pools, new_pools] if t is not None])
      pool_cache.update_cache(
        cache=next_pool_cache,
        prompts=next_cache_prompts,
      )
      emb_from_optional_text: EmbeddingFromOptionalText = lambda text: all_zeros_pool_ if text is None else pool_cache.get_by_prompt(text)

      make_get_embed_: Callable[[CFGPole], GetEmbed] = partial(
        make_get_embed,
        embedding_from_optional_text=emb_from_optional_text,
        interp_pole='from',
      )
      pool_sources: FloatTensor = embed_batch(
        batch_frames=batch_frames,
        get_cond_embed=make_get_embed_(cfg_pole='cond'),
        get_uncond_embed=make_get_embed_(cfg_pole='uncond') if cfg_scale > 1 else None,
      )
      pools.append(pool_sources)

    emb_from_optional_text: EmbeddingFromOptionalText = lambda text: all_zeros_hidden_states_ if text is None else emb_cache.get_by_prompt(text)

    make_make_get_embed: Callable[[InterpPole], Callable[[CFGPole], GetEmbed]] = lambda interp_pole: partial(
      make_get_embed,
      embedding_from_optional_text=emb_from_optional_text,
      interp_pole=interp_pole,
    )

    sources, targets = [embed_batch(
      batch_frames=batch_frames,
      get_cond_embed=make_get_embed_(cfg_pole='cond'),
      get_uncond_embed=make_get_embed_(cfg_pole='uncond') if cfg_scale > 1 else None,
    ) for make_get_embed_ in [
        make_make_get_embed(interp_pole) for interp_pole in ['from', 'to']
      ]
    ]

    quotients: List[float] = [
      frame.quotient if isinstance(frame, InterPrompt) else 0 for frame in batch_frames
    ]
    wants_lerp: List[bool] = [
      frame.strategy is InterpStrategy.Lerp if isinstance(frame, InterPrompt) else True for frame in batch_frames
    ]

    orig_dtype: DeviceType = sources.dtype
    # TODO: perhaps we should ban interpolating from an all-zeros source or to an all-zeros target?
    #       i.e. output all-zeros if either endpoint is all-zeros?
    interped: FloatTensor = interp_sources_to_targets(
      sources=sources.float(),
      targets=targets.float(),
      quotients=quotients,
      wants_lerp=wants_lerp,
    )
    interped_penult_hidden_states.append(interped.to(orig_dtype))
  
  base_embed: FloatTensor = cat(interped_penult_hidden_states, dim=-1)
  base_embedding_mask: BoolTensor = cat(embedding_masks, dim=-1)

  _, mask_vit_big_g = embedding_masks
  _, emb_vit_big_g = interped_penult_hidden_states
  _, input_ids_vit_big_g = input_ids_
  _, pools_vit_big_g = pools

  if use_refiner:
    refiner_embed: FloatTensor = emb_vit_big_g
    refiner_embedding_mask: BoolTensor = mask_vit_big_g

  if input_ids_vit_big_g is None:
    pooled_embed: FloatTensor = pools_vit_big_g
  else:
    emb_ixs_needing_interp: LongTensor = tensor([
      ix for ix, prompt in enumerate([
        *[frame.from_.uncond_prompt if isinstance(frame, InterPrompt) else None for frame in batch_frames] * (cfg_scale > 1),
        *[frame.from_.prompt if isinstance(frame, InterPrompt) else None for frame in batch_frames],
      ]) if prompt is not None
    ], dtype=torch.long, device=device)
    with (
      inference_mode(),
      to_device(vit_big_g.text_model.encoder.layers[-1], device) if swap_models else nullcontext(),
      sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext()
    ):
      last_hidden_state_vit_big_g: FloatTensor = forward_penultimate_hidden_state(
        penultimate_hidden_state=emb_vit_big_g.index_select(0, emb_ixs_needing_interp),
        final_encoder_layer=vit_big_g.text_model.encoder.layers[-1],
        input_ids_shape=input_ids_vit_big_g.size()
        # attention_mask=mask_vit_big_g,
      )
    with (
      inference_mode(),
      to_device(vit_big_g.text_model.final_layer_norm, device) if swap_models else nullcontext(),
      to_device(vit_big_g.text_projection, device) if swap_models else nullcontext(),
      sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext()
    ):
      pooled_interped_embeds: FloatTensor = pool_and_project_last_hidden_state(
        last_hidden_state=last_hidden_state_vit_big_g,
        final_layer_norm=vit_big_g.text_model.final_layer_norm,
        text_projection=vit_big_g.text_projection,
        input_ids=input_ids_vit_big_g,
      )

      pooled_embed: FloatTensor = pools_vit_big_g.scatter(
        0,
        emb_ixs_needing_interp.unsqueeze(-1).expand(-1, pools_vit_big_g.size(-1)),
        pooled_interped_embeds,
      )

  base_added_cond_kwargs = CondKwargs(
    text_embeds=pooled_embed,
    time_ids=base_time_ids.expand(base_embed.size(0), -1),
  )
  base_denoiser: Denoiser = base_denoiser_factory(
    cross_attention_conds=base_embed,
    added_cond_kwargs=base_added_cond_kwargs,
    cfg_scale=cfg_scale,
    # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
    # cross_attention_mask=base_embedding_mask.repeat(batch_size, 1),
  )
  if use_refiner:
    refiner_added_cond_kwargs = CondKwargs(
      text_embeds=pooled_embed,
      time_ids=refiner_time_ids.repeat_interleave(batch_size, 0) if cfg_scale > 1. else refiner_time_ids.expand(base_embed.size(0), -1),
    )
    refiner_denoiser: Denoiser = refiner_denoiser_factory(
      cross_attention_conds=refiner_embed,
      added_cond_kwargs=refiner_added_cond_kwargs,
      cfg_scale=refiner_cfg_scale,
      # TODO: check what mask is like. I assume Stability never trained on masked embeddings.
      # cross_attention_mask=refiner_embedding_mask.repeat(batch_size, 1),
    )

    base_id = IdentifiedDenoiser('base', base_denoiser)
    refiner_id = IdentifiedDenoiser('refiner', refiner_denoiser)
    def pick_delegate(sigma: float) -> IdentifiedDenoiser:
      return base_id if sigma > refine_from_sigma else refiner_id
    def on_delegate_change(from_: Optional[str], to: str) -> None:
      if from_ == 'base':
        base_unet.cpu()
      if to == 'base':
        base_unet.to(device)
      elif to == 'refiner':
        refiner_unet.to(device)

    denoiser = DispatchDenoiser(
      pick_delegate=pick_delegate,
      on_delegate_change=on_delegate_change if swap_models else None,
    )
  else:
    denoiser: Denoiser = base_denoiser

  # I've commented-out the noise sampler, because it's only usable with ancestral samplers.
  # for stable convergence: re-enable this if you decide to use sample_dpmpp_2m_sde.
  # for reproducibility: you'll want to use a batch-of-1, to have a seed-per-sample rather than per-batch.
  # noise_sampler = BrownianTreeNoiseSampler(
  #   latents,
  #   sigma_min=sigma_min,
  #   sigma_max=start_sigma,
  #   # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
  #   # I'm just re-using it because it's a convenient arbitrary number
  #   seed=seeds[0],
  # )

  unet_start: float = perf_counter()

  with inference_mode(), to_device(base_unet, device) if swap_models else nullcontext(), sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext():
    denoised_latents: FloatTensor = sample_dpmpp_2m(
      denoiser,
      latents,
      sigmas,
      # noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers such as sample_dpmpp_2m_sde
      # callback=callback,
    ).to(vae.dtype)

  if measure_sampling:
    torch.cuda.synchronize(device=device)

    unet_duration: float = perf_counter()-unet_start
    print(f'batch-of-{batch_size}: {unet_duration:.2f} secs')

  if swap_models:
    refiner_unet.cpu()

  denoised_latents = denoised_latents / vae.config.scaling_factor

  # cannot use flash attn because VAE decoder's self-attn has head_dim 512
  with inference_mode(), to_device(vae, device) if swap_models else nullcontext():#, sdp_kernel(enable_math=False) if torch.cuda.is_available() else nullcontext():
    decoder_out: DecoderOutput = vae.decode(denoised_latents.to(vae.dtype))

  sample: FloatTensor = decoder_out.sample.div(2).add(.5).clamp(0,1)
  for ix, (decoded, out_stem) in enumerate(zip(sample, out_stems)):
    # if you want lossless images: consider png
    out_name: str = join(out_dir, f'{out_stem}.jpg')
    img: Image = rgb_to_pil(decoded)
    img.save(out_name, subsampling=0, quality=95)
    print(f'Saved image: {out_name}')