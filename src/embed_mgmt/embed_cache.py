from torch import Tensor, FloatTensor
from typing import List, Dict, Optional, Generic, TypeVar
from src.embed_mgmt.mock_embed import mock_embed

TensorTy = TypeVar('TensorTy', bound=Tensor)

class EmbedCache(Generic[TensorTy]):
  prompts: List[str] = []
  cache: Optional[TensorTy] = None
  prompt_to_ix: Dict[str, int] = {}

  def fill_with_mocks(self) -> None:
    """make some recognisable pretend embeddings"""
    self.prompts = ['old', 'hello']
    self.cache = mock_embed(self.prompts)
    self.prompt_to_ix = self._get_embed_cache_prompt_to_ix(self.prompts)

  def get_cache_ix(self, prompt_text: str) -> Optional[int]:
    if prompt_text in self.prompt_to_ix:
      ix: int = self.prompt_to_ix[prompt_text]
      return ix
    return None

  def get_by_prompt(self, prompt_text: str) -> TensorTy:
    return self.cache[self.prompt_to_ix[prompt_text]]

  def update_cache(
    self,
    cache: TensorTy,
    prompts: List[str],
  ) -> None:
    self.cache = cache
    self.prompts = prompts
    self.prompt_to_ix: Dict[str, int] = EmbedCache._get_embed_cache_prompt_to_ix(prompts)

  @staticmethod
  def _get_embed_cache_prompt_to_ix(prompts: List[str]) -> Dict[str, int]:
    return { prompt: ix for ix, prompt in enumerate(prompts)}