from easing_functions import CubicEaseInOut
from typing import List, Callable, Set, Dict, NamedTuple, Optional
from torch import LongTensor, FloatTensor, BoolTensor, stack, tensor, cat, lerp, zeros
import torch
from functools import partial

from src.interpolation.intersperse_linspace import intersperse_linspace, InterpSpec
from src.interpolation.in_between import ManneredInBetweenParams
from src.interpolation.inter_prompt import InterPrompt
from src.interpolation.interp_strategy import InterpStrategy
from src.interpolation.slerp import slerp
from src.interpolation.interp_manner import InterpManner, QuotientModifier
from src.iteration.batched import batched
from src.sample_spec.prompts import CFGPrompts, PromptType
from src.sample_spec.sample_spec import SampleSpec
from src.embed_mgmt.make_get_embed import make_get_embed
from src.embed_mgmt.get_embed import GetEmbed, EmbeddingFromOptionalText
from src.embed_mgmt.get_prompt_text import InterpPole
from src.embed_mgmt.embed_batch import embed_batch
from src.embed_mgmt.mock_embed import mock_embed

force_zeros_for_empty_prompt = True
cfg_scale = 5.

# this is what we'll want eventually, but for now I'm testing the harder case (negative prompting) explicitly
# keyframes: List[PromptType] = [
#   CFGPrompts(
#     uncond_prompt=None if force_zeros_for_empty_prompt else '',
#     prompt=prompt,
#   ) if cfg_scale > 1 else NoCFGPrompts(
#     prompt=prompt,
#   ) for prompt in [
#     'hello',
#     'fellow',
#     'geese',
#   ]
# ]

keyframes: List[PromptType] = [
  CFGPrompts(
    uncond_prompt='goodbye',
    prompt='hello',
  ),
  CFGPrompts(
    uncond_prompt='adversary',
    prompt='fellow',
  ),
  CFGPrompts(
    uncond_prompt='taxes',
    prompt='geese',
  )
]

quotient_modifiers: List[QuotientModifier] = [lambda x:x, CubicEaseInOut()]

interp_specs: List[InterpSpec[InterpManner]] = [InterpSpec[InterpManner](
  steps=3,
  manner=InterpManner(
    quotient_modifier=modifier,
    strategy=InterpStrategy.Slerp,
  ),
) for _, modifier in zip(range(len(keyframes)-1), quotient_modifiers)]

def make_inbetween(params: ManneredInBetweenParams[PromptType, InterpManner]) -> InterPrompt[PromptType]:
  # apply easing function
  modified_quotient: float = params.manner.quotient_modifier(params.quotient)
  return InterPrompt[PromptType](
    from_=params.from_,
    to=params.to,
    quotient=modified_quotient,
    strategy=params.manner.strategy,
  )

frames: List[SampleSpec] = intersperse_linspace(
  keyframes=keyframes,
  make_inbetween=make_inbetween,
  interp_specs=interp_specs,
)

# friendly place to put a breakpoint
pass

max_batch_size=3

class EmbedSource(NamedTuple):
  from_cache: bool
  index: int

def get_embed_cache_prompt_to_ix(embed_cache_prompts: List[str]) -> Dict[str, int]:
  return { prompt: ix for ix, prompt in enumerate(embed_cache_prompts)}

all_zeros_embed: Optional[FloatTensor] = zeros((2,), dtype=torch.float32) if force_zeros_for_empty_prompt else None

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
    sample_prompts: List[PromptType] = [frame.from_, frame.to] if isinstance(frame, InterPrompt) else [frame]
    for prompt in sample_prompts:
      for prompt_text in [*[prompt.uncond_prompt] * isinstance(prompt, CFGPrompts), prompt.prompt]:
        if prompt_text is None:
          continue
        prompt_source: EmbedSource = register_prompt_text(prompt_text)
        from_cache, index = prompt_source
        if from_cache:
          retained_embed_ixs.add(index)
  
  new_encoded: Optional[LongTensor] = mock_embed(prompt_texts_ordered) if prompt_texts_ordered else None

  retained_embeds: Optional[LongTensor] = None if not retained_embed_ixs or embed_cache is None else embed_cache.index_select(0, tensor(list(retained_embed_ixs), dtype=torch.int64))

  assert retained_embeds is not None or new_encoded is not None
  embed_cache = cat([t for t in [retained_embeds, new_encoded] if t is not None])

  embed_cache_prompts: List[str] = [prompt for ix, prompt in enumerate(embed_cache_prompts) if ix in retained_embed_ixs] + prompt_texts_ordered
  embed_cache_prompt_to_ix: Dict[str, int] = get_embed_cache_prompt_to_ix(embed_cache_prompts)

  embedding_from_optional_text: EmbeddingFromOptionalText = lambda text: all_zeros_embed if text is None else embed_cache[embed_cache_prompt_to_ix[text]]

  make_make_get_embed: Callable[[InterpPole], GetEmbed] = lambda interp_pole: partial(
    make_get_embed,
    embedding_from_optional_text=embedding_from_optional_text,
    interp_pole=interp_pole,
  )

  sources, targets = [embed_batch(
    batch_frames=batch_frames,
    get_cond_embed=make_get_embed_(cfg_pole='cond'),
    get_uncond_embed=make_get_embed_(cfg_pole='uncond') if cfg_scale > 1 else None,
  ) for make_get_embed_ in [
      make_make_get_embed(interp_pole) for interp_pole in ['from', 'to']
    ]
  ]

  quotients: FloatTensor = tensor(
    [frame.quotient if isinstance(frame, InterPrompt) else 0 for frame in batch_frames],
    dtype=sources.dtype,
  ).unsqueeze(-1).repeat(2, 1)
  wants_lerp: BoolTensor = tensor([
    frame.strategy is InterpStrategy.Lerp if isinstance(frame, InterPrompt) else True for frame in batch_frames
  ]).unsqueeze(-1).repeat(2, 1)

  slerped: FloatTensor = slerp(
    sources,
    targets,
    quotients,
  )
  lerped: FloatTensor = lerp(
    sources,
    targets,
    quotients,
  )
  interped: FloatTensor = lerped.where(wants_lerp, slerped)
  pass