from typing import Dict, Any
import torch
from torch import inference_mode, FloatTensor
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import DecoderOutput, AutoencoderKLOutput, DiagonalGaussianDistribution
from PIL import Image
import numpy as np
from typing import Literal
from enum import Enum

from src.attn.null_attn_processor import NullAttnProcessor
from src.attn.natten_attn_processor import NattenAttnProcessor
from src.attn.qkv_fusion import fuse_vae_qkv

device = torch.device('cuda')

class VAEChoice(Enum):
  Ollin = 'ollin'
  SDXL = 'sdxl'
  Flux = 'flux'

vae_choice = VAEChoice.Flux
match vae_choice:
  case VAEChoice.Ollin:
    vae_descriptor = 'madebyollin/sdxl-vae-fp16-fix'
    vae_kwargs: Dict[str, Any] = {
      'torch_dtype': torch.float16,
    }
  case VAEChoice.SDXL:
    vae_descriptor = 'stabilityai/stable-diffusion-xl-base-0.9'
    vae_kwargs: Dict[str, Any] = {
      'variant': 'fp16',
      'subfolder': 'vae',
      'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    }
  case VAEChoice.Flux:
    vae_descriptor = 'black-forest-labs/FLUX.1-schnell'
    vae_kwargs: Dict[str, Any] = {
      'subfolder': 'vae',
      'torch_dtype': torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    }

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  vae_descriptor,
  use_safetensors=True,
  **vae_kwargs,
)

attn_impl: Literal['natten', 'null', 'original'] = 'natten'
match(attn_impl):
  case 'natten':
    fuse_vae_qkv(vae)
    # NATTEN seems to output identical output to global self-attention at kernel size 17
    # even kernel size 3 looks good (not identical, but very close).
    # I haven't checked what's the smallest kernel size that can look identical.
    # should save you loads of memory and compute
    # (neighbourhood attention costs do not exhibit quadratic scaling with sequence length)
    vae.set_attn_processor(NattenAttnProcessor(kernel_size=17))
  case 'null':
    for attn in [*vae.encoder.mid_block.attentions, *vae.decoder.mid_block.attentions]:
      # you won't be needing these
      del attn.to_q, attn.to_k
    # doesn't mix information between tokens via QK similarity. just projects every token by V and O weights.
    # looks alright, but is by no means identical to global self-attn.
    vae.set_attn_processor(NullAttnProcessor())
  case 'original':
    # leave it as global self-attention
    pass
  case _:
    raise ValueError('you are spicier than I anticipated')

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

save_image(sample, f'out.{vae_choice.value}.{attn_impl}.png')