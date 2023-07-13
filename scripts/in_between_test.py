from dataclasses import dataclass
from easing_functions import CubicEaseInOut
from typing import List, Callable, TypeAlias, Set, Dict, Optional

from src.interpolation.intersperse_linspace import intersperse_linspace, InterpSpec
from src.interpolation.in_between import ManneredInBetweenParams
from src.interpolation.inter_prompt import InterPrompt
from src.interpolation.interp_strategy import InterpStrategy
from src.iteration.batched import batched

QuotientModifier: TypeAlias = Callable[[float], float]

@dataclass
class InterpManner:
  quotient_modifier: QuotientModifier
  strategy: InterpStrategy

keyframes: List[str] = [
  'hello',
  'fellow',
  'geese',
]

quotient_modifiers: List[QuotientModifier] = [lambda x:x, CubicEaseInOut()]

interp_specs: List[InterpSpec[InterpManner]] = [InterpSpec[InterpManner](
  steps=3,
  manner=InterpManner(
    quotient_modifier=modifier,
    strategy=InterpStrategy.Slerp,
  ),
) for _, modifier in zip(range(len(keyframes)-1), quotient_modifiers)]

def make_inbetween(params: ManneredInBetweenParams[str, InterpManner]) -> InterPrompt:
  # apply easing function
  modified_quotient: float = params.manner.quotient_modifier(params.quotient)
  return InterPrompt(
    from_=params.from_,
    to=params.to,
    quotient=modified_quotient,
    strategy=params.manner.strategy,
  )

frames: List[str|InterPrompt] = intersperse_linspace(
  keyframes=keyframes,
  make_inbetween=make_inbetween,
  interp_specs=interp_specs,
)

# friendly place to put a breakpoint
pass



max_batch_size=2

encodings: Dict[str, str] = {'old': 'OLD'}

def register_prompt_text(prompt_text: str) -> None:
  if prompt_text not in prompt_text_to_ix:
    prompt_text_to_ix[prompt_text] = len(prompt_texts_ordered)
    prompt_texts_ordered.append(prompt_text)
  ix: int = prompt_text_to_ix[prompt_text]
  return ix

def encode(prompts: List[str]) -> List[str]:
  return [prompt.upper() for prompt in prompts]

for batch_ix, batch_frames in enumerate(batched(frames, max_batch_size)):
  prompt_text_to_ix: Dict[str, int] = {}
  prompt_texts_ordered: List[str] = []
  prompt_text_instance_ixs: List[List[int]] = []
  for frame in batch_frames:
    sample_prompt_text_instance_ixs: List[int] = []
    sample_prompts: List[str] = [frame.from_, frame.to] if isinstance(frame, InterPrompt) else [frame]
    for prompt in sample_prompts:
      prompt_ix: int = register_prompt_text(prompt)
      sample_prompt_text_instance_ixs.append(prompt_ix)
    prompt_text_instance_ixs.append(sample_prompt_text_instance_ixs)
  
  existing_prompts: Set[str] = encodings.keys() & prompt_text_to_ix.keys()
  new_prompts: Set[str] = prompt_text_to_ix.keys() - existing_prompts
  new_encoded: List[str] = encode(new_prompts)
  encodings = {
    **{key: encodings[key] for key in encodings if key in existing_prompts},
    **{key: value for key, value in zip(new_prompts, new_encoded)},
  }
  pass