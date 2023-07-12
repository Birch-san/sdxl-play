from typing import TypeVar, Generic, Protocol
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')
M = TypeVar('M')

@dataclass
class InBetweenParams(Generic[T]):
  from_: T
  to: T
  quotient: float

@dataclass
class ManneredInBetweenParams(InBetweenParams[T], Generic[T, M]):
  manner: M

class MakeInbetween(Protocol, Generic[T, M, U]):
  def __call__(self, params: ManneredInBetweenParams[T, M]) -> U: ...