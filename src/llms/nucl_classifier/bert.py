import random
from collections import defaultdict
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
	organism: str | None

NUCLEOTIDE_MAP = {
	"A": "[DNA_A]",
	"C": "[DNA_C]",
	"G": "[DNA_G]",
	"T": "[DNA_T]",
	"R": "[DNA_R]",
	"Y": "[DNA_Y]",
	"S": "[DNA_S]",
	"W": "[DNA_W]",
	"K": "[DNA_K]",
	"M": "[DNA_M]",
	"B": "[DNA_B]",
	"D": "[DNA_D]",
	"H": "[DNA_H]",
	"V": "[DNA_V]",
	"N": "[DNA_N]",
	"I": "[INTRON]",
	"E": "[EXON]",
	"U": "[DNA_UNKNOWN]"
}

class NuclBERT(BaseModel):
	max_length = 512
	flank_size = 16
	records_per_sequence = 250
	num_labels = 3
	
	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self.model = BertForSequenceClassification.from_pretrained(
			checkpoint,
			num_labels=self.num_labels
		)

		self.tokenizer = BertTokenizer.from_pretrained(
			checkpoint,
			do_lower_case=False
		)

		special_tokens = [
			"[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]",
			"[DNA_R]", "[DNA_Y]", "[DNA_S]", "[DNA_W]",
			"[DNA_K]", "[DNA_M]", "[DNA_B]", "[DNA_D]",
			"[DNA_H]", "[DNA_V]", "[DNA_N]", "[INTRON]",
			"[EXON]", "[DNA_PAD]", "[DNA_UNKNOWN]", "[DNA_INVALID]"]
		self.tokenizer.add_tokens(special_tokens)

		self.tokenizer.add_special_tokens({
			"additional_special_tokens": [
				"<|SEQUENCE|>",
				"<|ORGANISM|>",
				"<|GENE|>",
				"<|FLANK_BEFORE|>",
				"<|FLANK_AFTER|>",
				"<|PREDICTED_BEFORE|>",
				"<|TARGET|>"
			]
		})

		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = BertForSequenceClassification.from_pretrained(checkpoint)
		self.tokenizer = BertTokenizer.from_pretrained(checkpoint)
	
	def _process_sequence(
		self,
		sequence: str
	) -> str:
		result = []
		for nucl in sequence.upper():
			token = NUCLEOTIDE_MAP.get(nucl, "[DNA_INVALID]")
			result.append(token)
		return "".join(result)
	
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
			return "[EXON]"
		elif target == 1:
			return "[INTRON]"
		else:
			return "[DNA_UNKNOWN]"

	def build_input(
		self,
		sequence: str,
		target: str | None = None,
		organism: str | None = None
	) -> Input:
		return {
			"sequence": sequence,
			"target": target,
			"organism": organism
		}

	def _build_input(
		self,
		sequence: str,
		flank_before: str,
		flank_after: str,
		target: str | None = None,
		organism: str | None = None
	) -> tuple[str, int | None]:
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

		label = torch.tensor([target], dtype=torch.long)

		return (
			encoded_input["input_ids"].squeeze(0),
			encoded_input["attention_mask"].squeeze(0),
			label
		)
	
	def _prepare_dataset(
		self,
		data: list[Input]
	) -> Dataset:
		tokenized_dataset = []
		per_class = max(1, self.records_per_sequence // self.num_labels)
		
		for register in tqdm(data):
			sequence = register["sequence"]
			target = register["target"]
			organism = register["organism"]

			if target == None:
				raise MissingEssentialProp("Target missing")

			class_counts = defaultdict(int)

			indices = list(range(len(sequence)))
			random.shuffle(indices)

			for i in indices:
				cropped_target = target[i]
				label = self._process_target(cropped_target)

				if class_counts[label] >= per_class:
					continue

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

				tokenized_input = self._tokenize_for_training(
					sentence=sentence,
					target=label_id
				)

				input_ids, attention_mask, labels = tokenized_input

				tokenized_dataset.append({
					"input_ids": input_ids,
					"attention_mask": attention_mask,
					"labels": labels
				})

				class_counts[label] += 1

				if all(class_counts[c] >= per_class for c in range(self.num_labels)):
					break

		return Dataset.from_list(tokenized_dataset)

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
			per_device_eval_batch_size=params.batch_size,
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
				
				flank_before = self._process_sequence(flank_before)
				flank_after = self._process_sequence(flank_after)

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