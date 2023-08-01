from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
import torch
from torch import Generator
from PIL import Image
from typing import List

# scheduler args documented here:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L98
scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(
  'Birchlabs/waifu-diffusion-xl-unofficial',
  subfolder='scheduler',
  # sde-dpmsolver++ is very new. if your diffusers version doesn't have it: use 'dpmsolver++' instead.
  algorithm_type='sde-dpmsolver++',
  solver_order=2,
  # solver_type='heun' may give a sharper image. Cheng Lu reckons midpoint is better.
  solver_type='midpoint',
  use_karras_sigmas=True,
)

pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
  'Birchlabs/waifu-diffusion-xl-unofficial',
  scheduler=scheduler,
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant='fp16'
)
pipe.to('cuda')

# prompt = 'masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck'
prompt = 'pixiv, masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt,'
negative_prompt = 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name'

for seed in range(100, 130):
  out: StableDiffusionXLPipelineOutput = pipe.__call__(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=12.,
    original_size=(4096, 4096),
    target_size=(1024, 1024),
    height=1024,
    width=1024,
    generator=Generator().manual_seed(seed),
  )

  images: List[Image.Image] = out.images
  img, *_ = images

  img.save(f'out_wdxl/1_{seed}_waifu.png')