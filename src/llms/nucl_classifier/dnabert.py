import random
from typing import TypedDict, cast

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

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
	) -> None:
		self.max_length = max_length

		super().__init__(
			checkpoint=checkpoint,
			log_level=log_level,
			seed=seed
		)	
	
	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self.model = BertForSequenceClassification.from_pretrained(
			checkpoint,
			num_labels=self.num_labels,
			trust_remote_code=True
		)
		self.tokenizer = BertTokenizer.from_pretrained(checkpoint)

	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = BertForSequenceClassification.from_pretrained(
			checkpoint,
			num_labels=self.num_labels
		)
		self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
	
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
		)
		processed_sequence += self._process_sequence(sequence)
		processed_sequence += self._process_sequence(flank_after)

		return {
			"sequence": sequence,
			"target": target
		}
	

		output = f"<|SEQUENCE|>{self._process_sequence(sequence)}"
		
		output += f"<|FLANK_BEFORE|>{self._process_sequence(flank_before)}"
		
		output += f"<|FLANK_AFTER|>{self._process_sequence(flank_after)}"
		
		if organism:
			output += f"<|ORGANISM|>{organism[:10].lower()}"
		
		output += "<|TARGET|>"

		label = None
		if target:
			label = self._process_target(target)
		
		return output, label

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
			sentence,
			return_tensors="pt"
		)

		label = torch.tensor(target, dtype=torch.long)

		return (
			encoded_input["input_ids"].squeeze(0),
			encoded_input["attention_mask"].squeeze(0),
			label
		)
	
	def _prepare_dataset(
		self,
		data: list[Input]
	) -> Dataset:
		dataset = []
		for register in tqdm(data):
			sequence = register["sequence"]
			target = register["target"]
			organism = register["organism"]

			if target == None:
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
					target=cropped_target,
					organism=organism
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

				dataset.append(sample)

		return Dataset.from_list(dataset)

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
		organism = data["organism"]

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
					flank_after=flank_after,
					organism=organism
				)

				tokenized_input = self._tokenize(sentence)

				input_ids, _ = tokenized_input

				outputs = self.model(
					input_ids=input_ids
				)
				pred_id = torch.argmax(outputs.logits, dim=-1).item()

				predicted += self._unprocess_target(int(pred_id))

			predicted = (
				predicted.replace("[EXON]", "E")
				.replace("[INTRON]", "I")
				.replace("[DNA_UNKNOWN]", "U")
			)
			return predicted