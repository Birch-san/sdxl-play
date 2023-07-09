from torch import Tensor, FloatTensor, arange
import torch
from torch.nn.functional import pad
from .device import DeviceType

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
  """One-dimensional linear interpolation for monotonically increasing sample
  points.

  Returns the one-dimensional piecewise linear interpolant to a function with
  given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

  Args:
    x: the :math:`x`-coordinates at which to evaluate the interpolated
        values.
    xp: the :math:`x`-coordinates of the data points, must be increasing.
    fp: the :math:`y`-coordinates of the data points, same length as `xp`.

  Returns:
      the interpolated values, same size as `x`.
  """
  m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
  b = fp[:-1] - (m * xp[:-1])

  indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
  indicies = torch.clamp(indicies, 0, len(m) - 1)

  return m[indicies] * x + b[indicies]

def get_diffusers_euler_discrete_schedule(
  steps: int,
  # you can derive this via get_sigmas(alphas_cumprod)
  all_sigmas: FloatTensor,
  device: DeviceType,
):
  """
  Euler Discrete schedule with
    timestep_spacing == "leading"
    interpolation_type == "linear"
    use_karras_sigmas == False
  """
  step_ratio: int = 1000 // steps
  # I'm confused about why they bother rounding integers, and why they add 1 *after* casting to float.
  # it kinda suggests step_ratio used to be a float, which feels to me like it would've made more sense.
  timesteps: FloatTensor = (arange(0, steps, device=device) * step_ratio).round().flip(-1).to(all_sigmas.dtype).add(1)
  sigmas: FloatTensor = interp(timesteps, arange(0, len(all_sigmas), device=device), all_sigmas)
  sigmas = pad(sigmas, (0, 1), mode='constant')
  return sigmas

def get_init_noise_sigma(
  proposed_start_sigma: FloatTensor
) -> FloatTensor:
  """
  Modifies sigmas[0] the same way diffusers' Euler Discrete schedule does.
  https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/schedulers/scheduling_euler_discrete.py#L186
  I don't understand the reasoning. it only changes the value slightly. if you discretized it: you'd end up back where you started.
  Maybe it's unneeded, but changes so little it's harmless?
  """
  return (proposed_start_sigma ** 2 + 1) ** 0.5