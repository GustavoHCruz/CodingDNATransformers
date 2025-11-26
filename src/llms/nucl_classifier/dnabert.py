import random
from typing import TypedDict, cast

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from llms.base import BaseModel
from schemas.train_params import TrainParams
from utils.exceptions import MissingEssentialProp


class Input(TypedDict):
	sequence: str
	target: str | None

class GenerateInput(TypedDict):
	sequence: str 

class NuclDNABERT(BaseModel):
	model = None
	tokenizer = None
	num_labels = 3

	def __init__(
		self,
		checkpoint: str | None = None,
		log_level="INFO",
		seed: int | None = None,
		max_length: int = 512,
		flank_size: int = 24
	) -> None:
		self.max_length = max_length
		self.flank_size = flank_size

		super().__init__(
			checkpoint=checkpoint,
			log_level=log_level,
			seed=seed
		)	
	
	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self.model = AutoModel.from_pretrained(
			checkpoint,
			num_labels=self.num_labels,
			trust_remote_code=True
		)
		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = AutoModel.from_pretrained(
			checkpoint,
			num_labels=self.num_labels
		)
		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	
	def _process_sequence(
		self,
		sequence: str
	) -> str:
		return sequence
	
	def _process_target(
		self,
		target: str
	) -> int:
		if target == "E":
			return 0
		if target == "I":
			return 1
		if target == "U":
			return 2
		raise ValueError("Could not find a valid label.")
	
	def _unprocess_target(
		self,
		target: int
	) -> str:
		if target == 0:
			return "E"
		elif target == 1:
			return "I"
		else:
			return "U"

	def build_input(
		self,
		sequence: str,
		target: str | None = None
	) -> Input:
		return {
			"sequence": sequence,
			"target": target
		}

	def _build_input(
		self,
		sequence: str,
		flank_before: str,
		flank_after: str,
		target: str | None = None
	) -> tuple[str, int | None]:
		processed_sequence = (
			f"{self._process_sequence(flank_before)}[SEP]"
			f"{self._process_sequence(sequence)}[SEP]"
			f"{self._process_sequence(flank_after)}"
		)

		label = None
		if target:
			label = self._process_target(target)

		return processed_sequence, label

	def _tokenize(
		self,
		input_text: str
	) -> tuple[torch.Tensor, torch.Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")

		tokenized = self.tokenizer(
			input_text,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		).to(self.model.device)

		input_ids = tokenized["input_ids"]
		input_ids = cast(torch.Tensor, input_ids)
		attention_mask = tokenized["attention_mask"]
		attention_mask = cast(torch.Tensor, attention_mask)

		return (input_ids, attention_mask)

	def _tokenize_for_training(
		self,
		sentence: str,
		target: int
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		encoded_input = self.tokenizer(
			sentence
		)

		input_ids = torch.tensor(encoded_input["input_ids"], dtype=torch.long)
		attention_mask = torch.tensor(encoded_input["attention_mask"], dtype=torch.bool)
		label = torch.tensor([target], dtype=torch.long)

		return input_ids, attention_mask, label
	
	def _prepare_dataset(
		self,
		dataset: list[Input]
	) -> Dataset:
		tokenized = []

		for data in tqdm(dataset):
			sequence = data["sequence"]
			target = data.get("target")

			if target is None:
				raise MissingEssentialProp("Target missing")

			indices = list(range(len(sequence)))
			random.shuffle(indices)

			for i in indices:
				cropped_target = target[i]

				flank_start = max(i - self.flank_size, 0)
				flank_end = min(i + self.flank_size, len(sequence))
				
				flank_before = sequence[flank_start:i]
				flank_after = sequence[i+1:flank_end]
				
				sentence, label_id = self._build_input(
					sequence=sequence[i],
					flank_before=flank_before,
					flank_after=flank_after,
					target=cropped_target
				)

				assert label_id is not None

				input_ids, attention_mask, labels = self._tokenize_for_training(
					sentence=sentence,
					target=label_id
				)

				sample = {
					"input_ids": input_ids,
					"attention_mask": attention_mask,
					"labels": labels
				}

				tokenized.append(sample)

		return Dataset.from_list(tokenized)

	def train(
		self,
		dataset: list[Input],
		params: TrainParams
	) -> None:
		if not self.model or not self.tokenizer:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		self._log("Preparing dataset...")
		data = self._prepare_dataset(dataset)
		self._log("Dataset prepared!")

		self._log(f"Dataset length: {len(data)}")
		
		args = TrainingArguments(
			num_train_epochs=params.epochs,
			optim=params.optim,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			lr_scheduler_type="cosine",
			save_strategy="no",
			logging_steps=params.logging_steps
		)

		if self.seed:
			args.seed = self.seed
		
		trainer = Trainer(
			model=self.model,
			train_dataset=data,
			args=args,
			data_collator=DataCollatorWithPadding(self.tokenizer)
		)

		self._log("Starting training...")

		trainer.train()

		self._log("Training complete. You may save the model for later usage.")

	def generate(
		self,
		data: Input
	) -> str:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		self.model.eval()

		sequence = data["sequence"]

		predicted = ""
		
		with torch.no_grad():
			for i, nucl in enumerate(sequence):	
				flank_start = max(i - self.flank_size, 0)
				flank_end = min(i + self.flank_size, len(sequence))
				
				flank_before = sequence[flank_start:i]
				flank_after = sequence[i+1:flank_end]

				sentence, _ = self._build_input(
					sequence=nucl,
					flank_before=flank_before,
					flank_after=flank_after
				)

				tokenized_input = self._tokenize(sentence)

				input_ids, _ = tokenized_input

				outputs = self.model(
					input_ids=input_ids
				)
				pred_id = torch.argmax(outputs.logits, dim=-1).item()

				predicted += self._unprocess_target(int(pred_id))

			return predicted