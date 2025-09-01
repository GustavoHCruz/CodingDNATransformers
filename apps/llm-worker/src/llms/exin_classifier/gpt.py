import random
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset
from llms.base import BaseModel
from schemas.train_params import TrainParams
from tqdm import tqdm
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from utils.data_collators import DataCollatorForFT
from utils.exceptions import MissingEssentialProp


class Input(TypedDict):
	sequence: str
	target: str | None
	organism: str | None
	gene: str | None
	before: str | None
	after: str | None
	hide_prob: float | None

class GenerateInput(TypedDict):
	sequence: str
	organism: str | None
	gene: str | None
	before: str | None
	after: str | None
	hide_prob: float | None

class ExInClassifierGPT(BaseModel):
	model: GPT2LMHeadModel | None = None
	tokenizer: GPT2Tokenizer | None = None
	max_length = 1024

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self.model = GPT2LMHeadModel.from_pretrained(checkpoint)

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
	
	def _unprocess_target(
		self,
		target: str
	) -> str:
		return target.replace("[", "").replace("]", "")
	
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
		sequence: str,
		target: str | None = None,
		organism: str | None = None,
		gene: str | None = None,
		before: str | None = None,
		after: str | None = None,
		hide_prob: float = 0.01
	) -> dict[Literal["partial", "complete"], str]:
			output = f"<|SEQUENCE|>{self._process_sequence(sequence)}\n"

			if organism:
				if random.random() > hide_prob:
					output += f"<|ORGANISM|>{organism[:10]}\n"

			if gene:
				if random.random() > hide_prob:
					output += f"<|GENE|>{gene[:10]}\n"
			
			if before:
				before = self._process_sequence(before)
				if random.random() > hide_prob:
					output += f"<|FLANK_BEFORE|>{before}\n"
			
			if after:
				after = self._process_sequence(after)
				if random.random() > hide_prob:
					output += f"<|FLANK_AFTER|>{after}\n"
			
			output += "<|TARGET|>"

			return {
				"partial": output,
				"complete": f"{output}{(self._process_target(target) if target else '')}"
			}

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
		input_text: str,
		expected_text: str
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
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

	def _prepare_dataset(
		self,
		dataset: list[Input]
	) -> Dataset:
		tokenized = []

		for data in tqdm(dataset):
			promptfied = self._build_input(
				sequence=data["sequence"],
				target=data.get("target"),
				organism=data.get("organism", ""),
				gene=data.get("gene", ""),
				before=data.get("before", ""),
				after=data.get("after", ""),
				hide_prob=data.get("hide_prob") or 0.01
			)
			tokenized_input = self._tokenize_for_training(
				input_text=promptfied["partial"],
				expected_text=promptfied["complete"]
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
			data_collator=DataCollatorForFT(self.tokenizer),
		)

		self._log("Starting training...")

		trainer.train()

		self._log("Training complete. You may save the model for later use.")

	def generate(
		self,
		input: GenerateInput
	) -> str:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		model_input = self._build_input(
			sequence=input["sequence"],
			organism=input.get("organism"),
			gene=input.get("gene"),
			before=input.get("before"),
			after=input.get("after"),
			hide_prob=input.get("hide_prob") or 0.01
		)

		input_ids, attention_mask = self._tokenize(model_input["partial"])

		self.model.eval()
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=1,
				pad_token_id=self.tokenizer.eos_token_id
			)
		
		return self._unprocess_target(self.tokenizer.decode(outputs[0][-1]))