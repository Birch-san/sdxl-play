import torch
from torch import FloatTensor, IntTensor
import PIL
from PIL import Image
from numpy.typing import NDArray

# yes this is just torchvision save_image without writing to a file
# sure woulda been nice if they'd factored any of this into functions
def rgb_to_pil(sample: FloatTensor) -> Image.Image:
  uint: IntTensor = sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)
  ndarr: NDArray = uint.numpy()
  im = Image.fromarray(ndarr)
  return im