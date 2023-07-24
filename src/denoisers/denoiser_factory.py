from torch import FloatTensor, BoolTensor
from typing import Generic, TypeVar, Protocol, Optional
from .denoiser_proto import Denoiser
from ..added_cond import CondKwargs

D = TypeVar('D', bound=Denoiser)

# for situations such as batching: we may know early most construction params for our Denoiser subclass,
# but we would get the sample-specific concerns (cond kwargs) later (e.g. per batch item).
# this lets us handle most of the construction concern early, and delay the remaining concerns for later.
class DenoiserFactory(Generic[D], Protocol):
  def __call__(
    self,
    delegate: Denoiser,
    cross_attention_conds: FloatTensor,
    added_cond_kwargs: CondKwargs,
    cfg_scale: float,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> D: ...

class DenoiserFactoryFactory(Generic[D], Protocol):
  def __call__(self, denoiser: Denoiser) -> DenoiserFactory[D]: ...