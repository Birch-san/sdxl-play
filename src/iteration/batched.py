from itertools import islice
from typing import Iterable, Iterator, List, Generator, TypeVar

T = TypeVar('T')

# https://github.com/python/cpython/issues/98363
def batched(iterable: Iterable[T], n: int) -> Generator[List[T], None, None]:
  "Batch data into lists of length n. The last batch may be shorter."
  # batched('ABCDEFG', 3) --> ABC DEF G
  if n < 1:
    raise ValueError('n must be >= 1')
  it: Iterator[T] = iter(iterable)
  while (batch := list(islice(it, n))):
    yield batch