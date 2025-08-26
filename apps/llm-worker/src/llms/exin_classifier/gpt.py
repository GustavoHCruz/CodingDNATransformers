import json
import os
import random
from typing import Literal, cast

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, TrainingArguments
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer import Trainer

from llms.base import BaseModel
from schemas.train_params import TrainParams
from utils.data_collators import DataCollatorForFT


class ExInClassifierGPT(BaseModel):
	model: GPT2LMHeadModel | None = None
	tokenizer: GPT2Tokenizer | None = None
	max_length = 1024

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
		tokenizer.pad_token = tokenizer.eos_token

		special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[EXON]", "[INTRON]"]
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

		tokenizer.add_eos_token = True

		tokenizer.pad_token = tokenizer.eos_token
		self.tokenizer = tokenizer

		self.model = GPT2LMHeadModel.from_pretrained(checkpoint)

		if self.model is None or self.tokenizer is None:
			self._log("Error trying to load the checkpoint.", "WARNING")
			return None
		
		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
		self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

		df = pd.read_csv(checkpoint)
		self.history = df.to_dict()

		with open("info.json", "w", encoding="utf-8") as f:
			self.info = json.load(f)

	def _process_sequence(
		self,
		sequence: str
	) -> str:
		return f"".join(f"[{nucl.upper()}]" for nucl in sequence)

	def _process_target(
		self,
		target: str
	) -> str:
		return f"[{target.upper()}]"
	
	def build_prompt(
		self,
		sequence: str,
		target: str | None = None,
		organism: str = '',
		gene: str = '',
		before: str = '',
		after: str = '',
		hide_prob: float = 0.1
	) -> dict[Literal["question", "answer"], str]:
			output = f"<|SEQUENCE|>{sequence}\n"

			if organism:
				if random.random() > hide_prob:
					output += f"<|ORGANISM|>{organism[:10]}\n"

			if gene:
				if random.random() > hide_prob:
					output += f"<|GENE|>{gene[:10]}\n"
			
			if before:
				if random.random() > hide_prob:
					output += f"<|FLANK_BEFORE|>{before}\n"
			
			if after:
				if random.random() > hide_prob:
					output += f"<|FLANK_AFTER|>{after}\n"
			
			output += "<|TARGET|>"

			return {
				"question": output,
				"answer": f"{output}{target}"
			}

	def _tokenize(
		self,
		input_text: str,
		expected_text: str
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer missing", "WARNING")
			return None
		
		encoded_input = self.tokenizer(
			input_text
		)
		encoded_expected = self.tokenizer(
			expected_text
		)

		input_ids = torch.tensor(encoded_expected["input_ids"], dtype=torch.long)
		attention_mask = torch.tensor(encoded_expected["attention_mask"], dtype=torch.bool)

		labels = torch.full_like(input_ids, -100)

		start = len(cast(list, encoded_input["input_ids"]))

		for i in range(start, len(input_ids)):
			if input_ids[i] != self.tokenizer.pad_token_id:
				labels[i] = input_ids[i]
		
		return input_ids, attention_mask, labels

	def load_data(
		self,
		data: dict[Literal["sequence", "target", "organism", "gene", "before", "after"], list[str]],
		hide_prob: float = 0.1
	) -> None:
		tokenized_data = []
		for sequence, target, organism, gene, before, after in zip(data["sequence"], data["target"], data["organism"], data["gene"], data["before"], data["after"]):
			promptfied = self.build_prompt(
				sequence=sequence,
				target=target,
				organism=organism,
				gene=gene,
				before=before,
				after=after,
				hide_prob=hide_prob
			)

			tokenized = self._tokenize(
				input_text=promptfied["question"],
				expected_text=promptfied["answer"]
			)

			if tokenized:
				input_ids, attention_mask, labels = tokenized
				tokenized_data.append({
					"input_ids": input_ids,
					"attention_mask": attention_mask,
					"labels": labels
				})

		self.train_dataset = Dataset.from_list(tokenized_data)

	def train(
		self,
		params: TrainParams
	) -> None:
		if not self.model or not self.tokenizer:
			self._log("Model or Tokenizer not found", "WARNING")
			return None
		
		args = TrainingArguments(
			num_train_epochs=params.epochs,
			optim=params.optim,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			lr_scheduler_type="cosine",
			save_strategy="no"
		)

		trainer = Trainer(
			model=self.model,
			train_dataset=self.train_dataset,
			args=args,
			data_collator=DataCollatorForFT(self.tokenizer),
		)

		trainer.train()

	def evaluate(
		
	)

import os

import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
	model_name: str,
	uuid: str,
	data_length: int,
	batch_size: int,
	seed: int
):
	y_true = []
	y_pred = []

	model.eval()
	with torch.no_grad():
		for batch in tqdm(dataloader):
			input_ids, attention_mask, label = [b.to(model.device) for b in batch.values()]

			responses = model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=1,
				pad_token_id=tokenizer.eos_token_id
			)

			for r, l in zip(responses, label):
				# Pegamos o token previsto (último token gerado)
				pred_token = r[-1].item()
				true_token = l[0].item()

				y_pred.append(pred_token)
				y_true.append(true_token)

	# Calcula métricas
	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
	rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
	f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

	print(f"Accuracy:  {acc:.4f}")
	print(f"Precision: {prec:.4f}")
	print(f"Recall:    {rec:.4f}")
	print(f"F1-score:  {f1:.4f}")
	print(f"F1-score:  {f1:.4f}")
	print(f"F1-score:  {f1:.4f}")
