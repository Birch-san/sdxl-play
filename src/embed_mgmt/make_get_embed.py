from typing import Protocol
from functools import partial

from .get_embed import EmbeddingFromOptionalText, GetEmbed, get_embed
from .get_prompt_text import GetPromptText, InterpPole, CFGPole, get_prompt_text

class MakeGetEmbed(Protocol):
  def __call__(self, interp_pole: InterpPole) -> GetEmbed: ...

def make_get_embed(
  embedding_from_optional_text: EmbeddingFromOptionalText,
  interp_pole: InterpPole,
  cfg_pole: CFGPole,
) -> GetEmbed:
  get_prompt_txt: GetPromptText = partial(
    get_prompt_text,
    interp_pole=interp_pole,
    cfg_pole=cfg_pole,
  )
  return partial(
    get_embed,
    embedding_from_optional_text=embedding_from_optional_text,
    get_prompt_text=get_prompt_txt,
  )