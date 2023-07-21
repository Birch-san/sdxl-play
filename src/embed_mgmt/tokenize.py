from typing import NamedTuple, List
from logging import getLogger, Logger
from transformers import CLIPTokenizer, BatchEncoding
from torch import LongTensor, BoolTensor, tensor
import torch

from ..device import DeviceType

logger: Logger = getLogger(__file__)

class TokenizerOutput(NamedTuple):
  input_ids: LongTensor
  attention_mask: BoolTensor

def tokenize(
  tokenizer: CLIPTokenizer,
  prompts: List[str],
  device: DeviceType = torch.device('cpu'),
) -> TokenizerOutput:
  tokenized: BatchEncoding = tokenizer(
    prompts,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_overflowing_tokens=True,
  )
  overflows: List[List[int]] = tokenized.data['overflowing_tokens']
  overflows_decoded: List[str] = tokenizer.batch_decode(overflows)
  num_truncateds: List[int] = tokenized.data['num_truncated_tokens']
  for prompt_ix, (overflow_decoded, num_truncated) in enumerate(zip(overflows_decoded, num_truncateds)):
    if num_truncated > 0:
      logger.warning(f"Prompt {prompt_ix} will be truncated, due to exceeding tokenizer's length limit by {num_truncated} tokens. Overflowing portion of text was: <{overflow_decoded}>")

  input_ids: LongTensor = tensor(tokenized.input_ids, device=device)
  attention_mask: BoolTensor = tensor(tokenized.attention_mask, device=device, dtype=torch.bool)

  return TokenizerOutput(
    input_ids=input_ids,
    attention_mask=attention_mask,
  )