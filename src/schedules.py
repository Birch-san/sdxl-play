import torch
from dataclasses import dataclass
from enum import Enum, auto
from torch import Tensor, tensor
from .device import DeviceType

@dataclass
class KarrasScheduleParams:
  steps: int
  sigma_max: Tensor
  sigma_min: Tensor
  rho: float

class KarrasScheduleTemplate(Enum):
  # aggressively short sigma schedule; for cheap iteration
  Prototyping = auto()
  # cheap but reasonably good results; for exploring seeds to pick one to subsequently master in more detail
  Searching = auto()
  # higher quality, but still not too expensive
  Mastering = auto()
  # try to match Diffusers results
  Science = auto()
  # high-quality, for not-potato PC
  CudaMastering = auto()
  # increase rho slightly to include a sigmas 0.5692849159240723
  CudaMasteringMaximizeRefiner = auto()
  # silly number of steps, for scientific demonstrations
  Overkill = auto()

def get_template_schedule(
  template: KarrasScheduleTemplate,
  model_sigma_min: Tensor,
  model_sigma_max: Tensor,
  device: DeviceType,
  dtype: torch.dtype,
) -> KarrasScheduleParams:
  match(template):
    case KarrasScheduleTemplate.Prototyping:
      return KarrasScheduleParams(
        steps=5,
        sigma_max=tensor(7.0796, device=device, dtype=dtype),
        sigma_min=tensor(0.0936, device=device, dtype=dtype),
        rho=9.
      )
    case KarrasScheduleTemplate.Searching:
      return KarrasScheduleParams(
        steps=8,
        sigma_max=model_sigma_max,
        sigma_min=tensor(0.0936, device=device, dtype=dtype),
        rho=7.
      )
    case KarrasScheduleTemplate.Mastering:
      return KarrasScheduleParams(
        steps=15,
        sigma_max=model_sigma_max,
        sigma_min=model_sigma_min,
        rho=7.
      )
    case KarrasScheduleTemplate.CudaMastering:
      return KarrasScheduleParams(
        steps=25,
        sigma_max=model_sigma_max,
        sigma_min=model_sigma_min,
        rho=7.
      )
    case KarrasScheduleTemplate.CudaMasteringMaximizeRefiner:
      return KarrasScheduleParams(
        steps=25,
        sigma_max=model_sigma_max,
        sigma_min=model_sigma_min,
        rho=7.4
      )
    case KarrasScheduleTemplate.Science:
      # Diffusers EulerDiscreteScheduler#set_timesteps with timestep_spacing == 'leading'
      # does a weird thing where they don't start at timestep 999, but rather timestep (1000-(1000//steps))
      #   timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1]
      #   sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
      # but starting from that lower sigma_max does indeed seem to look pretty good
      return KarrasScheduleParams(
        steps=25,
        sigma_max=tensor(13.120415687561035, device=device, dtype=dtype),
        sigma_min=tensor(0.04131447896361351, device=device, dtype=dtype),
        rho=7.
      )
    case KarrasScheduleTemplate.Overkill:
      # this is for making pleasing animations of the denoising process
      return KarrasScheduleParams(
        steps=200,
        sigma_max=model_sigma_max,
        sigma_min=model_sigma_min,
        rho=7.
      )
    case _:
      raise ValueError(f"never heard of a {template} KarrasScheduleTemplate")