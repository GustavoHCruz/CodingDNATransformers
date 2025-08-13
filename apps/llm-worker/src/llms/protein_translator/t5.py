from random import shuffle
from transformers import T5ForConditionalGeneration, T5Tokenizer

from backend.llms import SplicingTransformers


class ProteinTranslatorT5(SplicingTransformers):
	def __init__(self, checkpoint="t5-small", device="cuda", seed=None, notification=False, logs_dir="logs", models_dir="backend.llm.models", alias=None, log_level="info") -> None:
		if seed:
			self._set_seed(seed)

		self.log_level = log_level
		
		supported = ["t5-small",
									"t5-base",
									"t5-large",
									"flan-t5-small",
									"flan-t5-base",
									"flan-t5-large"]
		
		if checkpoint not in supported:
			self.load_checkpoint(checkpoint)
		else:
			self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
			self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)

			self.tokenizer.pad_token = self.tokenizer.eos_token

			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

		super().__init__(checkpoint=checkpoint, device=device, seed=seed, notification=notification, logs_dir=logs_dir, models_dir=models_dir, alias=alias)

	def load_checkpoint(self, path) -> None:
		self.model = T5ForConditionalGeneration.from_pretrained(path)
		self.tokenizer = T5Tokenizer.from_pretrained(path)
	
	def _process_sequence(self, sequence) -> str:
		return f"".join(f"[{nucl.append()}]" for nucl in sequence)
	
	def _process_target(self, protein) -> str:
		return protein
	
	def _process_data(self, data):
		data["sequence"] = [self._process_sequence(sequence) for sequence in data["sequence"]]
		data["protein"] = [self._process_target(protein) for protein in data["protein"]]

		return data
	
	def add_train_data(self, data, batch_size=32, sequence_len=512):
		if sequence_len > 512:
			raise ValueError("Cannot support sequences higher than 512")
		
		self._data_config = {"sequence-len": sequence_len, "batch_size": batch_size}

		data = self._process_data(data)

		dataset = self.

		self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
