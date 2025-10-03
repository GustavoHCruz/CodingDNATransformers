import random
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertTokenizer,
                          DataCollatorWithPadding)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from llms.base import BaseModel
from schemas.train_params import TrainParams
from utils.exceptions import MissingEssentialProp


class Input(TypedDict):
	sequence: str
	target: str | None
	organism: str | None
	gene: str | None
	before: str | None
	after: str | None
	hide_prob: float | None

class ExInClassifierBERT(BaseModel):
	model: BertForSequenceClassification | None = None
	tokenizer: BertTokenizer | None = None
	max_length = 512

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self.model = BertForSequenceClassification.from_pretrained(
			checkpoint,
			num_labels=2
		)
		tokenizer = BertTokenizer.from_pretrained(
			checkpoint,
			do_lower_case=False
		)

		special_tokens = [
			"[A]", "[C]", "[G]", "[T]", "[R]",
			"[Y]", "[S]", "[W]", "[K]", "[M]",
			"[B]", "[D]", "[H]", "[V]", "[N]"
		]
		tokenizer.add_tokens(special_tokens)

		tokenizer.add_special_tokens({
			"additional_special_tokens": [
				"<|SEQUENCE|>",
				"<|ORGANISM|>",
				"<|GENE|>",
				"<|FLANK_BEFORE|>",
				"<|FLANK_AFTER|>",
				"<|TARGET|>"
			]
		})

		tokenizer.add_eos_tokens = True

		tokenizer.pad_tokens = tokenizer.eos_token
		self.tokenizer = tokenizer

		if self.model is None or self.tokenizer is None:
			self._log("Error trying to load the checkpoint.", "WARNING")
			return None
		
		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
	
	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = BertForSequenceClassification.from_pretrained(
			checkpoint
		)
		self.tokenizer = BertTokenizer.from_pretrained(
			checkpoint,
			num_labels=2
		)

	def _process_sequence(
		self,
		sequence: str
	) -> str:
		return f"".join(f"[{nucl.upper()}]" for nucl in sequence)
	
	def _process_target(
		self,
		target: str
	) -> Literal[0, 1]:
		if target == "EXON":
			return 0
		if target == "INTRON":
			return 1
		raise ValueError("Could not find a valid label.")
	
	def _unprocess_target(
		self,
		target: int
	) -> str:
		if target == 0:
			return "EXON"
		return "INTRON"
	
	def build_input(
		self,
		sequence: str,
		target: str | None = None,
		organism: str | None = None,
		gene: str | None = None,
		before: str | None = None,
		after: str | None = None,
		hide_prob: float = 0.01
	) -> Input:
		return {
			"sequence": sequence,
			"target": target,
			"organism": organism,
			"gene": gene,
			"before": before,
			"after": after,
			"hide_prob": hide_prob
		}
	
	def _build_input(
		self,
		data: Input,
		hide_prob: float = 0.01
	) -> tuple[str, int | None]:
		output = f"<|SEQUENCE|>{self._process_sequence(data["sequence"])}"

		if data["organism"]:
			if random.random() > hide_prob:
				output += f"<|ORGANISM|>{data["organism"][:10].lower()}"

		if data["gene"]:
			if random.random() > hide_prob:
				output += f"<|GENE|>{data["gene"][:10].lower()}"
		
		if data["before"]:
			before = self._process_sequence(data["before"])
			if random.random() > hide_prob:
				output += f"<|FLANK_BEFORE|>{before}"
		
		if data["after"]:
			after = self._process_sequence(data["after"])
			if random.random() > hide_prob:
				output += f"<|FLANK_AFTER|>{after}"
		
		output += "<|TARGET|>"

		label = None
		if data["target"]:
			label = self._process_target(data["target"])

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
			sentence, target = self._build_input(data)

			if target is None:
				raise ValueError("Target is missing.")

			tokenized_input = self._tokenize_for_training(
				sentence=sentence,
				target=target
			)

			input_ids, attention_mask, labels = tokenized_input
			tokenized.append({
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"labels": labels
			})

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

		args = TrainingArguments(
			num_train_epochs=params.epochs,
			optim=params.optim,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			lr_scheduler_type="cosine",
			save_strategy="no",
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
		
		sentence, _ = self._build_input(data)

		input_ids, _ = self._tokenize(sentence)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model(
				input_ids=input_ids
			)
			pred_id = torch.argmax(outputs.logits, dim=-1).item()
		
		return self._unprocess_target(int(pred_id))