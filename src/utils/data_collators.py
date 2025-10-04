import torch
from torch import Tensor
from transformers.data.data_collator import DataCollatorWithPadding


class DataCollatorForFT(DataCollatorWithPadding):
	def __call__(self, features) -> dict[str, Tensor]:
		labels = [f["labels"] for f in features] if "labels" in features[0] else None
		
		features_no_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]

		batch = super().__call__(features_no_labels)

		if labels is not None:
			labels_tensors = [torch.tensor(l, dtype=torch.long) for l in labels]
			batch["labels"] = torch.nn.utils.rnn.pad_sequence(
				labels_tensors,
				batch_first=True,
				padding_value=-100
			)
		
		return batch