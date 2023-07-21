import torch
from torch import BoolTensor, FloatTensor, LongTensor, arange
from torch.nn import LayerNorm, Linear
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, _make_causal_mask, _expand_mask
from typing import Optional, Tuple

def forward_penultimate_hidden_state(
  penultimate_hidden_state: FloatTensor,
  final_encoder_layer: CLIPEncoderLayer,
  input_ids_shape: torch.Size,
  attention_mask: Optional[BoolTensor] = None,
) -> FloatTensor:
  """
  Args:
    penultimate_hidden_state = text_encoder(…, return_dict=True).hidden_states[-2]
    final_encoder_layer = text_encoder.text_model.encoder.layers[-1]
    input_ids_shape = input_ids.size()
  Returns:
    Equivalent to text_encoder(…, return_dict=True).hidden_states[-1]
  """
  # text_encoder.text_model.encoder.layers[-1](encoder_out.hidden_states[-2], attention_mask=None, causal_attention_mask=_make_causal_mask(input_ids.size(), encoder_out.hidden_states[-2].dtype, device=encoder_out.hidden_states[-2].device))[0].allclose(encoder_out.hidden_states[-1])
  if attention_mask is not None:
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    attention_mask = _expand_mask(attention_mask, penultimate_hidden_state.dtype)
  causal_attention_mask: FloatTensor = _make_causal_mask(
    input_ids_shape=input_ids_shape,
    dtype=penultimate_hidden_state.dtype,
    device=penultimate_hidden_state.device,
  )
  layer_out: Tuple[FloatTensor] = final_encoder_layer(
    penultimate_hidden_state,
    attention_mask=attention_mask,
    causal_attention_mask=causal_attention_mask,
  )
  last_hidden_states, *_ = layer_out
  return last_hidden_states

def pool_and_project_last_hidden_state(
  # by this I mean hidden_states[-1], not last_hidden_state, which has already been layernormed
  last_hidden_state: FloatTensor,
  final_layer_norm: LayerNorm,
  text_projection: Linear,
  input_ids: LongTensor,
):
  """
  Args:
    last_hidden_state = text_encoder(…, return_dict=True).hidden_states[-1]
    final_layer_norm = text_encoder.text_model.final_layer_norm
    text_projection = text_encoder.text_projection
  """
  # text_encoder.text_model.final_layer_norm(encoder_out.hidden_states[-1]).allclose(encoder_out.last_hidden_state)
  normed: FloatTensor = final_layer_norm(last_hidden_state)
  # this is just saying "select the EOS token embed from each sample"
  pooled_output: FloatTensor = normed[
    # could we just use : here instead of an arange?
    arange(normed.shape[0], device=normed.device),
    # argmax looks for the highest token ID in the vocabulary, which we are assuming is the EOS token ID
    # (this will not work if you've expanded the vocabulary)
    input_ids.to(dtype=torch.int, device=normed.device).argmax(dim=-1),
  ]
  projected_output: FloatTensor = text_projection(pooled_output)
  return projected_output