from dataclasses import dataclass
from easing_functions import CubicEaseInOut
from typing import List, Callable, TypeAlias

from src.interpolation.intersperse_linspace import intersperse_linspace, InterpSpec
from src.interpolation.in_between import ManneredInBetweenParams
from src.interpolation.inter_prompt import InterPrompt
from src.interpolation.interp_strategy import InterpStrategy

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
  steps=5,
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

cond_linspace: List[str|InterPrompt] = intersperse_linspace(
  keyframes=keyframes,
  make_inbetween=make_inbetween,
  interp_specs=interp_specs,
)

# friendly place to put a breakpoint
pass