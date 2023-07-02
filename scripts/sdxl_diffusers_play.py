from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline #, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.vae import DecoderOutput
import torch
from torch import FloatTensor, no_grad
from torch.backends.cuda import sdp_kernel
from PIL import Image
from typing import List
from torchvision.utils import save_image

vae: AutoencoderKL = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae.to(device='cuda')#, dtype=torch.bfloat16)
vae.enable_slicing()

pipe: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
  # vae=vae,
  # I downloaded this previously using `use_auth_token=HF_TOKEN` but I will not be checking the HF token into this repo.
  local_files_only=True,
)
pipe.to('cuda')

prompt = "An astronaut riding a green horse"

seed=42
torch.manual_seed(seed)
with torch.inference_mode(), sdp_kernel(enable_math=False):
  base_images: FloatTensor = pipe(prompt=prompt, output_type="latent").images
base_images_scaled: FloatTensor = base_images / vae.config.scaling_factor


with torch.inference_mode():#, sdp_kernel(enable_math=False):
  decoder_out: DecoderOutput = vae.decode(base_images_scaled.to(vae.dtype))
  sample: FloatTensor = decoder_out.sample#.clone()
  sample = ((sample / 2) + 0.5).clamp(0,1)

  for ix, decoded in enumerate(sample):
    save_image(decoded, f'out/base_{ix}_{prompt}.{seed}.png')

pipe: StableDiffusionXLImg2ImgPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-refiner-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
  # vae=vae,
  # I downloaded this previously using `use_auth_token=HF_TOKEN` but I will not be checking the HF token into this repo.
  local_files_only=True,
  )
# 
pipe.to('cuda')#, dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()

torch.manual_seed(seed)
base_image, *_ = base_images
with torch.inference_mode():#, sdp_kernel(enable_math=False):
  refined_images: List[Image.Image] = pipe(prompt=prompt, image=base_image).images

for ix, image in enumerate(refined_images):
  image.save(f'out/refined_{ix}_{prompt}.{seed}.png')

# leaving this commented-out in case I can find a way to access the refined Unet output directly (the pipeline currently decodes it for you)
# refined_images_scaled: FloatTensor = refined_images / vae.config.scaling_factor
# image, *_ = images

# with no_grad():#, sdp_kernel(enable_math=False):
#   decoder_out: DecoderOutput = vae.decode(refined_images_scaled.to(vae.dtype))
#   sample: FloatTensor = decoder_out.sample
#   for ix, decoded in enumerate(sample):
#     save_image(decoded, f'out/refined_{ix}.png')