import torch
from transformers import StoppingCriteria


class StopOnToken(StoppingCriteria):
  def __init__(
    self,
    stop_token_ids: list[int],
  ) -> None:
    self.stop_token_ids = torch.tensor(stop_token_ids)

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
    last_token = input_ids[0, -1]
    stop_tokens = self.stop_token_ids.to(input_ids.device)

    return (last_token == stop_tokens).any() # type: ignore