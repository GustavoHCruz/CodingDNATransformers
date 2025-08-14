import csv
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from math import ceil
from typing import Any, Generator, Literal, cast

import torch
from accelerate import Accelerator
from config import SHARED_DIR, STORAGE_DIR
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.optimization import get_scheduler
from utils import load_model, save_model, set_seed
from schemas import TrainParams

class ExInClassifierGPT():
	model: GPT2LMHeadModel | None = None
	tokenizer: GPT2Tokenizer | None = None
	max_length = 512

	def _load_tokenizer(
		self,
		checkpoint: str
	) -> None:
		tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
		tokenizer.pad_token = tokenizer.eos_token

		special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[EXON]", "[INTRON]"]
		tokenizer.add_tokens(special_tokens)

		tokenizer.add_special_tokens({
			"additional_special_tokens": ["<|SEQUENCE|>", "<|ORGANISM|>", "<|GENE|>", "<|FLANK_BEFORE|>", "<|FLANK_AFTER|>", "<|TARGET|>"]
		})

		tokenizer.add_eos_token = True

		tokenizer.pad_token = tokenizer.eos_token

	def _load_model(
		self,
		checkpoint: str
	) -> None:
		self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
		self._load_tokenizer(
			checkpoint=checkpoint
		)
		if self.tokenizer is None:
			return None
		
		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		self._load_model(
			checkpoint=checkpoint
		)

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
	
	def build_prompt(
		self,
		sequence: str,
		target: str | None = None,
		organism: str = '',
		gene: str = '',
		before: str = '',
		after: str = '',
		hide_prob: float = 0.1
	) -> str:
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

			if target is None:
				return output

			return f"{output}{target}"

	def _tokenize(
		self,
		prompt: str,
		prompt_with_target: str
	) -> dict[Literal["input_ids", "attention_mask", "labels"], torch.Tensor] | None:
		if not self.tokenizer:
			return None
		
		prompt_encoded = self.tokenizer(prompt)
		prompt_with_target_encoded = self.tokenizer(
			prompt_with_target,
			truncation=True,
			padding="max_length",
			max_length=self.max_length
		)

		input_ids = prompt_with_target_encoded["input_ids"]
		input_ids = cast(list, input_ids)
		attention_mask = prompt_with_target_encoded["attention_mask"]

		labels = [-100] * len(input_ids)
		start = len(cast(list, prompt_encoded["input_ids"]))

		for i in range(start, len(input_ids)):
			if input_ids[i] != self.tokenizer.pad_token_id:
				labels[i] = input_ids[i]
		
		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
			"labels": torch.tensor(labels, dtype=torch.long)
		}

	def load_data(
		self,
		data: dict[Literal["sequence", "target", "organism", "gene", "before", "after"], list[str]],
		hide_prob: float = 0.1
	) -> None:
		tokenized_data = []
		for sequence, target, organism, gene, before, after in zip(data["sequence"], data["target"], data["organism"], data["gene"], data["before"], data["after"]):
			partial, full = self._promptfy(
				sequence=sequence,
				target=target,
				organism=organism,
				gene=gene,
				before=before,
				after=after,
				hide_prob=hide_prob
			)

			tokenized = self._tokenize(partial, full)

			tokenized_data.append(tokenized)

		self.data = tokenized_data
	
	def train(
		self,
		params: TrainParams
	) -> None:
		

class DNADatasetFinetune(IterableDataset):
		def __init__(
			self,
			csv_path: str,
			tokenizer,
			dataset_total_length: int,
			feat_hide_prob: float,
			flanks_size: int = 25,
			sequence_max_length: int = 512,
		) -> None:
			self.csv_path = csv_path
			self.tokenizer = tokenizer
			self.max_length = sequence_max_length + flanks_size * 2 + 20
			self._length = dataset_total_length
			self.feat_hide_prob = feat_hide_prob

		def __len__(self):
			return self._length
		
		def __iter__(self) -> Generator[dict[str, torch.Tensor], Any, None]:
			with open(self.csv_path, newline='') as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					sequence = process_sequence(row["sequence"])
					target = process_target(row["target"])
					organism = row["organism"]
					gene = row["gene"]
					flank_before = row["flankBefore"]
					flank_after = row["flankAfter"]

					partial, full = promptfy(
						sequence=sequence,
						target=target,
						organism=organism,
						gene=gene,
						flank_before=flank_before,
						flank_after=flank_after,
						hide_prob=self.feat_hide_prob,
					)

					partial_encoded = self.tokenizer(partial)
					full_encoded = self.tokenizer(
						full,
						truncation=True,
						padding="max_length",
						max_length=self.max_length
					)

					input_ids = full_encoded["input_ids"]
					attention_mask = full_encoded["attention_mask"]

					labels = [-100] * len(input_ids)
					start = min(len(partial_encoded["input_ids"]), len(input_ids))

					for i in range(start, len(input_ids)):
						if input_ids[i] != self.tokenizer.pad_token_id:
							labels[i] = input_ids[i]

					yield {
						"input_ids": torch.tensor(input_ids),
						"attention_mask": torch.tensor(attention_mask),
						"labels": torch.tensor(labels)
					}

class FinetuneDataCollator:
	def __init__(self, tokenizer) -> None:
		self.tokenizer = tokenizer
		self.pad_token_id = tokenizer.pad_token_id
	
	def __call__(self, batch) -> dict[str, torch.Tensor]:
		input_ids = [example["input_ids"] for example in batch]
		attention_mask = [example["attention_mask"] for example in batch]
		labels = [example["labels"] for example in batch]

		input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
		labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

		return {
			"input_ids": input_ids_padded,
			"attention_mask": attention_mask_padded,
			"labels": labels_padded
		}

def create_model(
	checkpoint: str,
	name: str,
	uuid: str,
	is_child: bool = False
) -> None:
	if is_child:
		parent_checkpoint = os.path.join(STORAGE_DIR, "models", name)
		model = AutoModelForCausalLM.from_pretrained(
			parent_checkpoint,
			low_cpu_mem_usage=False
		)
		tokenizer = AutoTokenizer.from_pretrained(parent_checkpoint)
	else:
		model = AutoModelForCausalLM.from_pretrained(checkpoint)
		tokenizer = AutoTokenizer.from_pretrained(checkpoint)

		special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[EXON]", "[INTRON]"]
		tokenizer.add_tokens(special_tokens)

		tokenizer.add_special_tokens({
			"additional_special_tokens": ["<|SEQUENCE|>", "<|ORGANISM|>", "<|GENE|>", "<|FLANK_BEFORE|>", "<|FLANK_AFTER|>", "<|TARGET|>"]
		})

		tokenizer.add_eos_token = True

		tokenizer.pad_token = tokenizer.eos_token
		model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
	
	output_path = os.path.join(STORAGE_DIR, "models", name)
	model.save_pretrained(output_path)
	tokenizer.save_pretrained(output_path)

def train_model(
	model_name: str,
	uuid: str,
	data_length: int,
	epochs: int,
	batch_size: int,
	gradient_accumulation: int,
	lr: float,
	warmup_ratio: float,
	feat_hide_prob: float,
	seed: int
) -> None:
	set_seed(seed)

	accelerator = Accelerator()
	is_main_process = accelerator.is_main_process
	num_gpus = accelerator.num_processes

	model, tokenizer = load_model(model_name)

	data_path = os.path.join(SHARED_DIR, "temp", uuid)

	dataset = DNADatasetFinetune(
		csv_path=data_path+".csv",
		tokenizer=tokenizer,
		dataset_total_length=data_length,
		feat_hide_prob=feat_hide_prob
	)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		collate_fn=FinetuneDataCollator(tokenizer)
	)

	optimizer = AdamW(model.parameters(), lr=lr)
	num_training_steps = epochs * len(dataloader)
	num_warmup_steps = int(warmup_ratio * num_training_steps)

	lr_scheduler = get_scheduler(
		name="cosine",
		optimizer=optimizer,
		num_warmup_steps=num_warmup_steps,
		num_training_steps=num_training_steps
	)

	model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
		model, optimizer, dataloader, lr_scheduler
	)

	dataloader_len_local = len(dataloader)

	history = {"epoch": [], "time": [], "train_loss": [], "lr": []}
	start_time = time.time()

	num_update_steps_per_epoch = ceil(dataloader_len_local / gradient_accumulation)
	max_train_steps = epochs * num_update_steps_per_epoch

	global_step = 0
	model.train()
	for epoch in range(epochs):
		train_loss = 0.0

		accumulated_loss = 0.0
		for batch_idx, batch in enumerate(dataloader):
			outputs = model(**batch)
			loss = outputs.loss / gradient_accumulation

			accelerator.backward(loss)
			accumulated_loss += loss.item()

			if (batch_idx + 1) % gradient_accumulation == 0 or (batch_idx + 1 == dataloader_len_local):
				accelerator.clip_grad_norm_(model.parameters(), max_norm=0.5)

				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

				loss_val = loss.item()
				train_loss += loss_val
				global_step += 1

				steps_in_epoch = ceil(dataloader_len_local / gradient_accumulation)
				current_epoch_fraction = epoch + (global_step % steps_in_epoch) / steps_in_epoch

				current_lr = lr_scheduler.get_last_lr()[0]

				if is_main_process:
					history["epoch"].append(round(current_epoch_fraction, 2))
					history["train_loss"].append(accumulated_loss)
					history["lr"].append(current_lr)
					history["time"].append(time.time() - start_time)

				accumulated_loss = 0.0
	
	accelerator.wait_for_everyone()
	model = accelerator.unwrap_model(model)

	if is_main_process:
		save_model(model_name, model, tokenizer, history)
	
	accelerator.wait_for_everyone()
	accelerator.end_training()

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
	set_seed(seed)

	model, tokenizer = load_model(model_name)

	data_path = os.path.join(SHARED_DIR, "temp", uuid)

	dataset = DNADatasetFinetune(
		csv_path=data_path + ".csv",
		tokenizer=tokenizer,
		dataset_total_length=data_length,
		feat_hide_prob=0.0
	)
	dataloader = DataLoader(
			dataset=dataset,
			batch_size=batch_size,
			collate_fn=FinetuneDataCollator(tokenizer)
	)

	model.to("cuda")

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
