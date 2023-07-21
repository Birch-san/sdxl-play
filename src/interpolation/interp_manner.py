from dataclasses import dataclass
from typing import Callable, TypeAlias
from .interp_strategy import InterpStrategy

QuotientModifier: TypeAlias = Callable[[float], float]

@dataclass
class InterpManner:
  quotient_modifier: QuotientModifier
  strategy: InterpStrategy