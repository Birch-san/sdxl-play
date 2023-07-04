from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
import torch
from torch import FloatTensor
from torch.backends.cuda import sdp_kernel
from PIL import Image
from typing import List
from torchvision.utils import save_image
from torchvision.transforms import Resize, InterpolationMode

def round_down(num: int, divisor: int) -> int:
  return num - (num % divisor)

pipe: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
)
pipe.to('cuda')

area = 1024**2
# SDXL's demo code tries to round dimensions to nearest multiple of 64
width_px = round_down(1024, 64)
height_px = round_down(area//width_px, 64)
width_lt = width_px // pipe.vae_scale_factor
height_lt = height_px // pipe.vae_scale_factor

# Emad suggested the base model prefers 512x512. you can try that out by uncommenting the //2 lines, but I found 512x512 looks pretty bad
# base_width_px = round_down(width_px//2, 64)
# base_height_px = round_down(height_px//2, 64)
base_width_px = width_px
base_height_px = height_px

vae: AutoencoderKL = pipe.vae
vae.enable_slicing()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "90s anime sketch, girl wearing serafuku walking home, masterpiece, dramatic, wind"
# prompt = "90s anime promo art, girl wearing serafuku walking home, masterpiece, dramatic, wind"
# prompt = "An astronaut riding a green horse"
# prompt = "my anime waifu is so cute"

seed=0
torch.manual_seed(seed)
with torch.inference_mode(), sdp_kernel(enable_math=False):
  base_images: FloatTensor = pipe(prompt=prompt, output_type="latent", width=base_width_px, height=base_height_px).images
base_images_unscaled: FloatTensor = base_images / vae.config.scaling_factor

with torch.inference_mode():
  decoder_out: DecoderOutput = vae.decode(base_images_unscaled.to(torch.float16))
sample: FloatTensor = decoder_out.sample
sample = ((sample / 2) + 0.5).clamp(0,1)

for ix, decoded in enumerate(sample):
  save_image(decoded, f'out/base_{ix}_{prompt}.{seed}.png')

pipe: StableDiffusionXLImg2ImgPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-refiner-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
)
pipe.to('cuda')
# pipe.enable_model_cpu_offload()

if base_width_px != width_px or base_height_px != height_px:
  vae: AutoencoderKL = pipe.vae
  vae.enable_slicing()

  doubler = Resize((height_px, width_px), interpolation=InterpolationMode.BICUBIC, antialias=True)
  # base_images_resized = doubler(base_images)
  sample_resized: FloatTensor = doubler(sample)
  for ix, resized in enumerate(sample_resized):
    save_image(resized, f'out/base_{ix}_{prompt}_resized.{seed}.png')
  plusminus1 = (sample_resized * 2) - 1
  with torch.inference_mode():
    encoder_out: AutoencoderKLOutput = vae.to(torch.bfloat16).encode(plusminus1.to(torch.bfloat16))
  sample_resized_latent_dist: DiagonalGaussianDistribution = encoder_out.latent_dist
  gen: torch.Generator = torch.manual_seed(seed)
  sample_resized_latents: FloatTensor = sample_resized_latent_dist.sample(gen)
  sample_resized_latents_scaled: FloatTensor = sample_resized_latents * vae.config.scaling_factor

  base_image, *_ = sample_resized_latents_scaled
else:
  base_image, *_ = base_images

refiner_seed=1024
torch.manual_seed(refiner_seed)
strength=0.3
with torch.inference_mode():
  refined_images: List[Image.Image] = pipe(prompt=prompt, image=base_image, strength=strength).images

for ix, image in enumerate(refined_images):
  image.save(f'out/refined_{ix}_{prompt}.{seed}.s{refiner_seed}.str{strength:.1f}.png')