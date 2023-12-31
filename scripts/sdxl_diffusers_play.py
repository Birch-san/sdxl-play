from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
import torch
from torch import FloatTensor, Generator
from torch.backends.cuda import sdp_kernel
from PIL import Image
from typing import List, Callable
from torchvision.transforms import Resize, InterpolationMode
from os import makedirs, listdir
from os.path import join
import fnmatch
from pathlib import Path
from PIL import Image

from src.device import get_device_type, DeviceType
from src.rgb_to_pil import rgb_to_pil

device_type: DeviceType = get_device_type()
device = torch.device(device_type)

def round_down(num: int, divisor: int) -> int:
  return num - (num % divisor)

base: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
)
base.to(device)

area = 1024**2
# SDXL's demo code tries to round dimensions to nearest multiple of 64
width_px = round_down(1024, 64)
height_px = round_down(area//width_px, 64)
width_lt = width_px // base.vae_scale_factor
height_lt = height_px // base.vae_scale_factor

# Emad suggested the base model prefers 512x512. you can try that out by uncommenting the //2 lines, but I found 512x512 looks pretty bad
# base_width_px = round_down(width_px//2, 64)
# base_height_px = round_down(height_px//2, 64)
base_width_px = width_px
base_height_px = height_px

vae: AutoencoderKL = base.vae
vae.to(torch.bfloat16)
vae.enable_slicing()

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "astronaut meditating under waterfall, in swimming shorts"
# prompt = "90s anime sketch, girl wearing serafuku walking home, masterpiece, dramatic, wind"
# prompt = "90s anime promo art, girl wearing serafuku walking home, masterpiece, dramatic, wind"
prompt: str = 'photo of astronaut meditating under waterfall, in swimming shorts, breathtaking, 4k, dslr, cinematic'
# prompt = "An astronaut riding a green horse"
# prompt = "my anime waifu is so cute"

refine = False

if refine:
  # TODO: this probably brings in another copy of the VAE, so make them share
  refiner: StableDiffusionXLImg2ImgPipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
  )
  refiner.to('cuda')
  # refiner.enable_model_cpu_offload()

# set range to (start, start+1) to generate 1 image
for seed in range(42, 43):
  generator = Generator(device='cpu').manual_seed(seed)

  with torch.inference_mode(), sdp_kernel(enable_math=False):
    base_images: FloatTensor = base(
      prompt=prompt,
      output_type="latent",
      width=base_width_px,
      height=base_height_px,
      generator=generator,
    ).images
  base_images_unscaled: FloatTensor = base_images / vae.config.scaling_factor

  with torch.inference_mode():
    decoder_out: DecoderOutput = vae.decode(base_images_unscaled.to(torch.float16))
  sample: FloatTensor = decoder_out.sample
  sample = ((sample / 2) + 0.5).clamp(0,1)

  out_dir = 'out'
  makedirs(out_dir, exist_ok=True)

  out_imgs_unsorted: List[str] = fnmatch.filter(listdir(out_dir), f'*_*.*')
  get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
  out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
  out_imgs: List[str] = sorted(out_imgs_unsorted, key=out_keyer)
  next_ix = get_out_ix(Path(out_imgs[-1]).stem)+1 if out_imgs else 0
  out_stem: str = f'{next_ix:05d}_base_{prompt.split(",")[0]}_{seed}_diffu'

  for ix, decoded in enumerate(sample):
    out_name: str = join(out_dir, f'{out_stem}.jpg')
    img: Image = rgb_to_pil(decoded)
    img.save(out_name, subsampling=0, quality=95)
    print(f'Saved base image: {out_name}')

  if refine:
    # science code path: tests what happens if we generate small base image,
    # crudely upsample RGB, then img2img with refiner.
    # disabled by default, because unfortunately:
    # - base is bad at low-res images (forgot how to do 512 after it was finetuned to do 1024)
    # - strength=0.3 on refiner isn't enough to fill in the detail missing following a crude upsample
    if base_width_px != width_px or base_height_px != height_px:
      doubler = Resize((height_px, width_px), interpolation=InterpolationMode.BICUBIC, antialias=True)
      # base_images_resized = doubler(base_images)
      sample_resized: FloatTensor = doubler(sample)
      for ix, resized in enumerate(sample_resized):
        resized_stem = f'{out_stem}_upscaled'
        out_name: str = join(out_dir, f'{resized_stem}.jpg')
        img: Image = rgb_to_pil(resized)
        img.save(out_name, subsampling=0, quality=95)
        print(f'Saved upscaled base image: {out_name}')
      plusminus1 = (sample_resized * 2) - 1
      with torch.inference_mode():
        encoder_out: AutoencoderKLOutput = vae.encode(plusminus1.to(vae.dtype))
      sample_resized_latent_dist: DiagonalGaussianDistribution = encoder_out.latent_dist
      gen: torch.Generator = torch.manual_seed(seed)
      sample_resized_latents: FloatTensor = sample_resized_latent_dist.sample(gen)
      sample_resized_latents_scaled: FloatTensor = sample_resized_latents * vae.config.scaling_factor

      base_image, *_ = sample_resized_latents_scaled
    else:
      base_image, *_ = base_images

    # seed 42 seemed to be a pretty opinionated comic style. 1024 is a bit nothingy.
    refiner_seed=1024
    # it's very weird to have to specify a seed for the refiner.
    # I want to get away from img2img and just combine them into one pipeline with one noise schedule.
    torch.manual_seed(refiner_seed)
    strength=0.3
    with torch.inference_mode():
      refined_images: List[Image.Image] = refiner(prompt=prompt, image=base_image, strength=strength).images

    out_stem: str = f'{next_ix:05d}_refined_{prompt.split(",")[0]}_{seed}.s{refiner_seed}.str{strength:.1f}'

    for ix, image in enumerate(refined_images):
      out_name: str = join(out_dir, f'{out_stem}.jpg')
      img: Image = rgb_to_pil(decoded)
      img.save(out_name, subsampling=0, quality=95)
      print(f'Saved refined image: {out_name}')