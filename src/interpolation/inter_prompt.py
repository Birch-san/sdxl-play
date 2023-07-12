from dataclasses import dataclass
from .in_between import InBetweenParams
from .interp_strategy import InterpStrategy

@dataclass
class InterPrompt(InBetweenParams[str]):
  from_: str
  to: str
  quotient: float
  strategy: InterpStrategy