from typing import Literal, TypedDict

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from llms.base import BaseModel
from schemas.train_params import TrainParams
from utils.exceptions import MissingEssentialProp


class Input(TypedDict):
	sequence: str
	target: str | None

class GenerateInput(TypedDict):
	sequence: str

class ExInClassifierDNABERT(BaseModel):
	model = None
	tokenizer = None
	max_length = 512

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, trust_remote_code=True)
		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
		self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
		
	def _process_sequence(
		self,
		sequence: str
	) -> str:
		return sequence
	
	def _process_target(
		self,
		label: str
	) -> Literal[0, 1]:
		return 0 if label == "INTRON" else 1
	
	def _unprocess_target(
		self,
		target: int
	) -> str:
		if target == 0:
			return "INTRON"
		return "EXON"

	def build_input(
		self,
		sequence: str,
		target: str | None
	) -> Input:
		return {
			"sequence": sequence,
			"target": target
		}
	
	def _build_input(
		self,
		sequence: str,
		target: str | None = None
	) -> tuple[str, int | None]:
		label = None
		if target:
			label = self._process_target(target)
		
		return sequence, label
	
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
		attention_mask = tokenized["attention_mask"]

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
			sentence, target = self._build_input(
				sequence=data["sequence"],
				target=data.get("target")
			)

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
			data_collator=DataCollatorWithPadding(self.tokenizer),
		)

		self._log("Starting training...")

		trainer.train()

		self._log("Training complete. You may save the model for later usage.")

	def generate(
		self,
		input: GenerateInput
	) -> str:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		sentence, _ = self._build_input(
			sequence=input["sequence"]
		)

		input_ids, _ = self._tokenize(sentence)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model(
				input_ids=input_ids
			)
			pred_id = torch.argmax(outputs.logits, dim=-1).item()
		
		return self._unprocess_target(int(pred_id))