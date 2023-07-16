from torch import FloatTensor
from typing import Callable, Optional, TypeAlias, Protocol

from ..sample_spec.sample_spec import SampleSpec

EmbeddingFromOptionalText: TypeAlias = Callable[[Optional[str]], FloatTensor]

def get_embed(
  embedding_from_optional_text: EmbeddingFromOptionalText,
  get_prompt_text: Callable[[SampleSpec], Optional[str]],
  sample_spec: SampleSpec,
) -> str:
  prompt_text: Optional[str] = get_prompt_text(sample_spec=sample_spec)
  embed: FloatTensor = embedding_from_optional_text(prompt_text)
  return embed

class GetEmbed(Protocol):
  def __call__(self, sample_spec: SampleSpec) -> FloatTensor: ...

