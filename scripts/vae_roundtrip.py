from typing import Dict, Any
import torch
from torch import inference_mode, FloatTensor
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import DecoderOutput, AutoencoderKLOutput
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from PIL import Image
import numpy as np
from src.attn.natten_attn_processor import NattenAttnProcessor
from src.attn.qkv_fusion import fuse_vae_qkv

device = torch.device('cuda')

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
  'madebyollin/sdxl-vae-fp16-fix' if use_ollin_vae else 'stabilityai/stable-diffusion-xl-base-0.9',
  use_safetensors=True,
  **vae_kwargs,
)
fuse_vae_qkv(vae)
# you'll need a dev build of NATTEN to use kernel sizes as large as 17. otherwise you'll have to go down to 13.
vae.set_attn_processor(NattenAttnProcessor(kernel_size=17))
# vae.enable_slicing() # you probably don't need this any more
vae.eval().to(device)

def load_img(path) -> FloatTensor:
  image: Image.Image = Image.open(path).convert("RGB")
  w, h = image.size
  print(f"loaded input image of size ({w}, {h}) from {path}")
  img_arr: np.ndarray = np.array(image)
  del image
  img_tensor: FloatTensor = torch.from_numpy(img_arr).to(dtype=torch.float32)
  del img_arr
  # TODO: would contiguous() make it faster to convolve over this?
  img_tensor: FloatTensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
  img_tensor: FloatTensor = img_tensor / 127.5 - 1.0
  return img_tensor
img: FloatTensor = load_img('in.jpg')

generator = torch.Generator(device='cpu')
generator.manual_seed(42)

with inference_mode():
  encoded: AutoencoderKLOutput = vae.encode(img.to(device, vae.dtype))
  dist: DiagonalGaussianDistribution = encoded.latent_dist
  latents: FloatTensor = dist.sample(generator=generator)
  decoder_out: DecoderOutput = vae.decode(latents)

sample: FloatTensor = decoder_out.sample.div(2).add_(.5).clamp_(0,1)

save_image(sample, 'out.png')