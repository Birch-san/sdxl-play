from dataclasses import dataclass
from typing import TypeVar, Generic
from .in_between import InBetweenParams
from .interp_strategy import InterpStrategy

P = TypeVar('P')

@dataclass
class InterPrompt(InBetweenParams[P], Generic[P]):
  from_: P
  to: P
  quotient: float
  strategy: InterpStrategy