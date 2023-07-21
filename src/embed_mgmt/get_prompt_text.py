from typing import Callable, TypeAlias, Literal, Optional
from ..sample_spec.sample_spec import SampleSpec
from ..sample_spec.prompts import PromptType, CFGPrompts
from ..interpolation.inter_prompt import InterPrompt

InterpPole: TypeAlias = Literal['from', 'to']
CFGPole: TypeAlias = Literal['uncond', 'cond']

GetPromptText: TypeAlias = Callable[[SampleSpec], Optional[str]]
def get_prompt_text(
  sample_spec: SampleSpec,
  interp_pole: InterpPole,
  cfg_pole: CFGPole,
) -> Optional[str]:
  if isinstance(sample_spec, InterPrompt):
    prompt: PromptType = sample_spec.from_ if interp_pole == 'from' else sample_spec.to
  else:
    prompt: PromptType = sample_spec
  if cfg_pole == 'cond':
    return prompt.prompt
  assert isinstance(prompt, CFGPrompts)
  return prompt.uncond_prompt