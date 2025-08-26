import torch
from torch import Tensor
from transformers import DataCollatorWithPadding


class DataCollatorForFT(DataCollatorWithPadding):
	def __call__(self, features) -> dict[str, Tensor]:
		batch = super().__call__(features)

		if "labels" in features[0]:
			labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
			batch["labels"] = torch.nn.utils.rnn.pad_sequence(
				labels, batch_first=True, padding_value=-100
			)

		return batch