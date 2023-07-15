from dataclasses import dataclass
from easing_functions import CubicEaseInOut
from typing import List, Callable, TypeAlias, Set, Dict, NamedTuple, Optional
from torch import LongTensor, FloatTensor, full, stack, tensor, cat, lerp
import torch

from src.interpolation.intersperse_linspace import intersperse_linspace, InterpSpec
from src.interpolation.in_between import ManneredInBetweenParams
from src.interpolation.inter_prompt import InterPrompt
from src.interpolation.interp_strategy import InterpStrategy
from src.interpolation.slerp import slerp
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


unique_num = 0
def mock_embed(prompts: List[str]) -> LongTensor:
  global unique_num
  start_ix=unique_num
  unique_num=unique_num+len(prompts)
  return stack([
    # using float32 because this script runs on-CPU, and CPU doesn't implement a float16 lerp.
    # in reality we'll run text encoders in float16, so we'll get float16 embeds.
    # *however* we might want to cast-to-fp32 to do the slerp.
    full((2,), fill_value=fill_value, dtype=torch.float32) for fill_value in range(
      start_ix,
      unique_num,
    )
  ])

max_batch_size=3

class EmbedSource(NamedTuple):
  from_cache: bool
  index: int

def get_embed_cache_prompt_to_ix(embed_cache_prompts: List[str]) -> Dict[str, int]:
  return { prompt: ix for ix, prompt in enumerate(embed_cache_prompts)}

# make some recognisable pretend embeddings
# embed_cache_prompts: List[str] = ['old', 'hello']
# embed_cache: LongTensor = mock_embed(embed_cache_prompts)
# embed_cache_prompt_to_ix: Dict[str, int] = get_embed_cache_prompt_to_ix(embed_cache_prompts)
embed_cache_prompts: List[str] = []
embed_cache: Optional[LongTensor] = None
embed_cache_prompt_to_ix: Dict[str, int] = {}

def register_prompt_text(prompt_text: str) -> EmbedSource:
  if prompt_text in embed_cache_prompt_to_ix:
    ix: int = embed_cache_prompt_to_ix[prompt_text]
    return EmbedSource(from_cache=True, index=ix)
  if prompt_text not in prompt_text_to_ix:
    prompt_text_to_ix[prompt_text] = len(prompt_texts_ordered)
    prompt_texts_ordered.append(prompt_text)
  ix: int = prompt_text_to_ix[prompt_text]
  return EmbedSource(from_cache=False, index=ix)

for batch_ix, batch_frames in enumerate(batched(frames, max_batch_size)):
  prompt_text_to_source: Dict[str, EmbedSource] = {}
  prompt_text_to_ix: Dict[str, int] = {}
  prompt_texts_ordered: List[str] = []
  retained_embed_ixs: Set[int] = set()

  for frame in batch_frames:
    sample_prompts: List[str] = [frame.from_, frame.to] if isinstance(frame, InterPrompt) else [frame]
    for prompt in sample_prompts:
      prompt_source: EmbedSource = register_prompt_text(prompt)
      from_cache, index = prompt_source
      if from_cache:
        retained_embed_ixs.add(index)
  
  new_encoded: Optional[LongTensor] = mock_embed(prompt_texts_ordered) if prompt_texts_ordered else None

  retained_embeds: Optional[LongTensor] = None if not retained_embed_ixs or embed_cache is None else embed_cache.index_select(0, tensor(list(retained_embed_ixs), dtype=torch.int64))

  assert retained_embeds is not None or new_encoded is not None
  embed_cache = cat([t for t in [retained_embeds, new_encoded] if t is not None])

  embed_cache_prompts = [prompt for ix, prompt in enumerate(embed_cache_prompts) if ix in retained_embed_ixs] + prompt_texts_ordered
  embed_cache_prompt_to_ix: Dict[str, int] = get_embed_cache_prompt_to_ix(embed_cache_prompts)

  # [[frame.from_, frame.to] if isinstance(frame, InterPrompt) else [frame] for frame in batch_frames]
  # [[embed_cache_prompt_to_ix[prompt] for prompt in tup] for tup in [[frame.from_, frame.to] if isinstance(frame, InterPrompt) else [frame] for frame in batch_frames]]
  # [[embed_cache[embed_cache_prompt_to_ix[frame.from_]], embed_cache[embed_cache_prompt_to_ix[frame.to]]] if isinstance(frame, InterPrompt) else embed_cache[embed_cache_prompt_to_ix[frame]] for frame in batch_frames]
  # TODO: check strategy; use lerp when appropriate
  x: FloatTensor = [slerp(embed_cache[embed_cache_prompt_to_ix[frame.from_]], embed_cache[embed_cache_prompt_to_ix[frame.to]], frame.quotient) if isinstance(frame, InterPrompt) else embed_cache[embed_cache_prompt_to_ix[frame]] for frame in batch_frames]
  pass