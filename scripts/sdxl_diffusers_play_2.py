# approach taken by Zippy (EleutherAI)
import torch
from diffusers import DiffusionPipeline
from torchvision.utils import save_image
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from PIL import Image
from typing import List

base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
base.to("cuda")
refiner.to('cuda')

prompt = "An astronaut riding a green horse"
with torch.inference_mode():
  seed=42
  torch.manual_seed(seed)
  image = base(prompt=prompt, output_type="latent").images

  image_save = image.clone()
  image_save = base.vae.decode(image_save / base.vae.config.scaling_factor, return_dict=False)[0]
  image_save = ((image_save / 2) + 0.5).clamp(0,1)

  save_image(image_save,"out/zippy_base.jpg")

  images: List[Image.Image] = refiner(prompt=prompt, image=image).images

  for idx,image in enumerate(images):
    image.save(f"out/zippy_image_{idx}.jpg")