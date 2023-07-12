from dataclasses import dataclass

from src.interpolation.intersperse_linspace import intersperse_linspace, InterpSpec
from src.interpolation.in_between import ManneredInBetweenParams
from src.interpolation.inter_prompt import InterPrompt
from src.interpolation.interp_strategy import InterpStrategy

from typing import List

@dataclass
class InterpManner:
  strategy: InterpStrategy

keyframes: List[str] = [
  'hello',
  'fellow',
  'geese',
]

interp_specs: List[InterpSpec[InterpManner]] = [InterpSpec[InterpManner](
  steps=3,
  manner=InterpManner(
    strategy=InterpStrategy.Slerp,
  ),
)] * (len(keyframes)-1)

def make_inbetween(params: ManneredInBetweenParams[str, InterpManner]) -> InterPrompt:
  return InterPrompt(
    from_=params.from_,
    to=params.to,
    quotient=params.quotient,
    strategy=params.manner.strategy,
  )

cond_linspace: List[InterPrompt] = intersperse_linspace(
  keyframes=keyframes,
  make_inbetween=make_inbetween,
  interp_specs=interp_specs,
)

# friendly place to put a breakpoint
pass