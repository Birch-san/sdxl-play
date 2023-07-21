from torch import FloatTensor
from typing import Union, NamedTuple, Optional
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from .tokenize import TokenizerOutput

class EmbedderOutput(NamedTuple):
  penultimate_hidden_states: FloatTensor
  norm_pooled: Optional[FloatTensor]

def embed(
  text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection],
  tokenizer_output: TokenizerOutput,
):
  """
  Returns:
    penultimate hidden states
  """
  encoder_out: Union[BaseModelOutputWithPooling, CLIPTextModelOutput] = text_encoder(
    input_ids=tokenizer_output.input_ids,
    # don't pass the mask in; the result is horrible. I guess it was trained on pad tokens?
    # attention_mask=tokenizer_output.attention_mask,
    output_hidden_states=True,
    return_dict=True,
  )
  penultimate_hidden_states: FloatTensor = encoder_out.hidden_states[-2]
  if isinstance(encoder_out, CLIPTextModelOutput):
    norm_pooled: FloatTensor = encoder_out.text_embeds
  else:
    norm_pooled = None
  return EmbedderOutput(
    penultimate_hidden_states=penultimate_hidden_states,
    norm_pooled=norm_pooled,
  )

  
