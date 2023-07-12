from dataclasses import dataclass
from typing import List, Iterable, Generic
from itertools import chain, pairwise
import numpy as np

from .in_between import ManneredInBetweenParams, MakeInbetween, T, U, M

@dataclass
class InterpSpec(Generic[M]):
  steps: int
  manner: M

def intersperse_linspace(
  keyframes: List[T],
  make_inbetween: MakeInbetween[T, M, U],
  interp_specs: List[InterpSpec[M]],
) -> Iterable[T | U]:
  assert len(keyframes) > 1
  assert len(interp_specs) == len(keyframes)-1
  return (
    *chain(
      *(
        (
          pair[0],
          *(
            make_inbetween(
              ManneredInBetweenParams[T, M](
                from_=pair[0],
                to=pair[1],
                quotient=step,
                manner=interp_spec.manner,
              )
            ) for step in np.linspace(
                start=1/interp_spec.steps,
                stop=1,
                num=interp_spec.steps-1,
                endpoint=False,
            )
          )
        ) for pair, interp_spec in zip(pairwise(keyframes), interp_specs)
      )
    ),
    keyframes[-1]
  )