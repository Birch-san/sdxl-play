from typing import Dict, Any
import torch
from torch import inference_mode, FloatTensor
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import DecoderOutput, AutoencoderKLOutput
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from PIL import Image
import numpy as np
from typing import Literal
from time import perf_counter

from src.attn.null_attn_processor import NullAttnProcessor
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

natten_kernel=15
attn_impl: Literal['natten', 'null', 'original'] = 'natten'
match(attn_impl):
  case 'natten':
    fuse_vae_qkv(vae)
    # NATTEN seems to output identical output to global self-attention at kernel size 17
    # even kernel size 3 looks good (not identical, but very close).
    # I haven't checked what's the smallest kernel size that can look identical.
    # should save you loads of memory and compute
    # (neighbourhood attention costs do not exhibit quadratic scaling with sequence length)
    vae.set_attn_processor(NattenAttnProcessor(kernel_size=natten_kernel))
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

alloc = torch.cuda.memory_allocated(0)
print(f'{alloc/1024**2:.2f} MiB allocated')
alloc_plus_reserved = torch.cuda.memory_reserved(0)
print(f'{alloc_plus_reserved/1024**2:.2f} MiB allocated+reserved')

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
# img: FloatTensor = load_img('in.jpg').to(device, vae.dtype)
img: FloatTensor = load_img('out_hpf_interp_c3/00003_base_masterpiece_s42_t0.4285714626312256.jpg').to(device, vae.dtype)

generator = torch.Generator(device='cpu')
generator.manual_seed(42)

iterations = 1
batch_size = 1
img = img.expand(batch_size, *img.shape[1:])

print('initial:')
alloc = torch.cuda.memory_allocated(0)
print(f'{alloc/1024**2:.2f} MiB allocated')
alloc_plus_reserved = torch.cuda.memory_reserved(0)
print(f'{alloc_plus_reserved/1024**2:.2f} MiB allocated+reserved')

torch.cuda.synchronize()
tic = perf_counter()
with inference_mode():
  for _ in range(iterations):
    encoded: AutoencoderKLOutput = vae.encode(img)
    dist: DiagonalGaussianDistribution = encoded.latent_dist
    latents: FloatTensor = dist.sample(generator=generator)
    print('after encoding, sampling:')
    alloc = torch.cuda.memory_allocated(0)
    print(f'{alloc/1024**2:.2f} MiB allocated')
    alloc_plus_reserved = torch.cuda.memory_reserved(0)
    print(f'{alloc_plus_reserved/1024**2:.2f} MiB allocated+reserved')
    decoder_out: DecoderOutput = vae.decode(latents)
    torch.cuda.synchronize()
duration_secs = perf_counter() - tic
secs_per_sample = duration_secs/(iterations*batch_size)
samples_per_sec = secs_per_sample**-1
print(f'{iterations} batches of {batch_size}:')
print(f'{secs_per_sample:.2f} secs/sample')
print(f'{samples_per_sec:.2f} samples/sec')
alloc = torch.cuda.memory_allocated(0)
print(f'{alloc/1024**2:.2f} MiB allocated')
alloc_plus_reserved = torch.cuda.memory_reserved(0)
print(f'{alloc_plus_reserved/1024**2:.2f} MiB allocated+reserved')
# original
# 20 batches of 8:
# 0.26 secs/sample
# 3.87 samples/sec

sample: FloatTensor = decoder_out.sample[:1].div(2).add_(.5).clamp_(0,1)

save_image(sample, f'out_roundtrip/1.{attn_impl}{natten_kernel}.png')