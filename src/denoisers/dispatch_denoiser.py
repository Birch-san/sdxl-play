from dataclasses import dataclass, field
from typing import Protocol, Optional, NamedTuple
from torch import FloatTensor

from .denoiser_proto import Denoiser

class IdentifiedDenoiser(NamedTuple):
  name: str
  denoiser: Denoiser

class PickDenoiseDelegate(Protocol):
  def __call__(self, sigma: float) -> IdentifiedDenoiser: ...

class OnDelegateChange(Protocol):
  def __call__(self, from_: Optional[str], to: str) -> None: ...

@dataclass
class DispatchDenoiser(Denoiser):
  pick_delegate: PickDenoiseDelegate
  on_delegate_change: Optional[OnDelegateChange] = None
  current_delegate_name: Optional[str] = field(init=False, default=None)

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    name, denoiser = self.pick_delegate(sigma[0].item())
    if name != self.current_delegate_name:
      if self.on_delegate_change is not None:
        self.on_delegate_change(self.current_delegate_name, name)
      self.current_delegate_name = name
    denoised: FloatTensor = denoiser(noised_latents, sigma)
    return denoised