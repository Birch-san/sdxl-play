from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline #, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.vae import DecoderOutput
import torch
from torch import FloatTensor, no_grad
from torch.backends.cuda import sdp_kernel
# from PIL import Image
# from typing import List
from torchvision.utils import save_image

vae: AutoencoderKL = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae.to(device='cuda')#, dtype=torch.bfloat16)
# vae.enable_slicing()
# model = "stabilityai/your-stable-diffusion-model"
# pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

pipe: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
  # I downloaded this previously using `use_auth_code=HF_TOKEN` but I will not be checking the HF token into this repo.
  local_files_only=True,
)
pipe.to('cuda')

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

seed=42
torch.manual_seed(seed)
base_images: FloatTensor = pipe(prompt=prompt, output_type="latent").images
base_images_scaled: FloatTensor = base_images / vae.config.scaling_factor


with no_grad(), sdp_kernel(enable_math=False):
  decoder_out: DecoderOutput = vae.decode(base_images_scaled.to(vae.dtype))
  sample: FloatTensor = decoder_out.sample
  clamped: FloatTensor = sample.clamp(-1., 1.)
  for ix, decoded in enumerate(sample):
    save_image(decoded, f'out/base_{ix}.png')

# for ix, pil in enumerate(pils):
#   pil.save(f'out/base_{ix}.png')

pipe: StableDiffusionXLImg2ImgPipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-refiner-0.9",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16",
  # I downloaded this previously using `use_auth_code=HF_TOKEN` but I will not be checking the HF token into this repo.
  local_files_only=True,
  )
# pipe.enable_model_cpu_offload()

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

torch.manual_seed(seed)
base_image, *_ = base_images
refined_images: FloatTensor = pipe(prompt=prompt, image=base_image).images
# image, *_ = images

# image.save('out/refined.png')

with no_grad(), sdp_kernel(enable_math=False):
  decoder_out: DecoderOutput = vae.decode(refined_images.to(vae.dtype))
  for ix, decoded in enumerate(decoder_out.sample):
    save_image(decoded, f'out/refined_{ix}.png')