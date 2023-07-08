from torch import FloatTensor
from typing import TypedDict

class CondKwargs(TypedDict):
  text_embeds: FloatTensor
  time_ids: FloatTensor