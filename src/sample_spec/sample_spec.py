from typing import TypeAlias, Union
from .prompts import PromptType
from ..interpolation.inter_prompt import InterPrompt

SampleSpec: TypeAlias = Union[PromptType, InterPrompt[PromptType]]