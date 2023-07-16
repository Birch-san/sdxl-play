from dataclasses import dataclass
from typing import Optional, TypeAlias, Union

@dataclass
class CFGPrompts:
  # None indicates an all-zeros uncond (as opposed to an empty-string uncond). it doesn't disable CFG.
  uncond_prompt: Optional[str]
  prompt: str

@dataclass
class NoCFGPrompts:
  prompt: str

PromptType: TypeAlias = Union[CFGPrompts, NoCFGPrompts]