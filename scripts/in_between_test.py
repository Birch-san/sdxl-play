from dataclasses import dataclass
from easing_functions import CubicEaseInOut
from typing import Any, List, Callable, TypeAlias, Set, Dict, NamedTuple, Optional, Union, TypeAlias, Literal, Protocol
from torch import LongTensor, FloatTensor, BoolTensor, full, stack, tensor, cat, lerp, zeros
import torch
from functools import partial

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

@dataclass
class CFGPrompts:
  # None indicates an all-zeros uncond (as opposed to an empty-string uncond). it doesn't disable CFG.
  uncond_prompt: Optional[str]
  prompt: str

@dataclass
class NoCFGPrompts:
  prompt: str

PromptType: TypeAlias = Union[CFGPrompts, NoCFGPrompts]

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

SampleSpec: TypeAlias = Union[PromptType, InterPrompt[PromptType]]

frames: List[SampleSpec] = intersperse_linspace(
  keyframes=keyframes,
  make_inbetween=make_inbetween,
  interp_specs=interp_specs,
)

# friendly place to put a breakpoint
pass


unique_num = 1 # reserve all-zeros embed for uncond
def mock_embed(prompts: List[str]) -> LongTensor:
  global unique_num
  start_ix=unique_num
  unique_num=unique_num+len(prompts)
  return stack([
    # using float32 because this script runs on-CPU, and CPU doesn't implement a float16 lerp.
    # in reality we'll run text encoders in float16, so we'll get float16 embeds.
    # *however* we might want to cast-to-fp32 to do the slerp.
    tensor((fill_value, 0, 1), dtype=torch.float32) for fill_value in range(
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

InterpPole: TypeAlias = Literal['from', 'to']
CFGPole: TypeAlias = Literal['uncond', 'cond']

GetPromptText: TypeAlias = Callable[[SampleSpec], Optional[str]]
def get_prompt_text(
  sample_spec: SampleSpec,
  interp_pole: InterpPole,
  cfg_pole: CFGPole,
) -> Optional[str]:
  if isinstance(sample_spec, InterPrompt):
    prompt: PromptType = sample_spec.from_ if interp_pole == 'from' else sample_spec.to
  else:
    prompt: PromptType = sample_spec
  if cfg_pole == 'cond':
    return prompt.prompt
  assert isinstance(prompt, CFGPrompts)
  return prompt.uncond_prompt

EmbeddingFromOptionalText: TypeAlias = Callable[[Optional[str]], FloatTensor]

def get_embed(
  embedding_from_optional_text: EmbeddingFromOptionalText,
  get_prompt_text: Callable[[SampleSpec], Optional[str]],
  sample_spec: SampleSpec,
) -> str:
  prompt_text: Optional[str] = get_prompt_text(sample_spec=sample_spec)
  embed: FloatTensor = embedding_from_optional_text(prompt_text)
  return embed

class GetEmbed(Protocol):
  def __call__(self, sample_spec: SampleSpec) -> FloatTensor: ...

class MakeGetEmbed(Protocol):
  def __call__(self, interp_pole: InterpPole) -> GetEmbed: ...

def embed_batch(
  batch_frames: List[Union[PromptType, InterPrompt[PromptType]]],
  get_cond_embed: GetEmbed,
  get_uncond_embed: Optional[GetEmbed],
) -> FloatTensor:
  unconds: List[FloatTensor] = [] if get_uncond_embed is None else [
    get_uncond_embed(sample_spec=frame) for frame in batch_frames
  ]
  conds: List[FloatTensor] = [
    get_cond_embed(sample_spec=frame) for frame in batch_frames
  ]
  return stack([*unconds, *conds])

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

  make_get_embed: Callable[[InterpPole, CFGPole], GetEmbed] = lambda interp_pole, cfg_pole: partial(
    get_embed,
    embedding_from_optional_text=embedding_from_optional_text,
    get_prompt_text=partial(
      get_prompt_text,
      interp_pole=interp_pole,
      cfg_pole=cfg_pole,
    )
  )
  make_make_get_embed: Callable[[InterpPole], GetEmbed] = lambda interp_pole: partial(make_get_embed, interp_pole=interp_pole)

  sources, targets = [embed_batch(
    batch_frames=batch_frames,
    get_cond_embed=bound_make_get_embed(cfg_pole='cond'),
    get_uncond_embed=bound_make_get_embed(cfg_pole='uncond') if cfg_scale > 1 else None,
  ) for bound_make_get_embed in [
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