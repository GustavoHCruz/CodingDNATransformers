import csv
import os
import time
from datetime import datetime
from math import ceil
from typing import Any, Generator

import pandas as pd
import torch
from accelerate import Accelerator
from config import SHARED_DIR, STORAGE_DIR
from llms.utils import set_seed
from redis_service import redis_service
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_scheduler

valid_dna = set("ACGTURYSWKMBDHVN")
valid_prot = set("ACDEFGHIKLMNPQRSTVWY*X")

def process_sequence(sequence: str) -> str:
	return "".join(f"[DNA_{nucl.upper()}]" for nucl in sequence if nucl.upper() in valid_dna)

def process_target(target: str) -> str:
	target = target + "*"
	target = target[:target.find("*") + 1]
	return "".join(f"[PROT_{prot.upper()}]" for prot in target if prot.upper() in valid_prot)

def promptfy(sequence: str, target: str) -> tuple[str, str]:
	return f"<|DNA|> {sequence} <|PROTEIN|>", f"<|DNA|> {sequence} <|PROTEIN|> {target}"

class DNADatasetFinetune(IterableDataset):
	def __init__(self, csv_path: str, tokenizer, dataset_total_length: int, sequence_max_length=1024) -> None:
		self.csv_path = csv_path
		self.tokenizer = tokenizer
		self.max_length = sequence_max_length
		self._length = dataset_total_length
	
	def __len__(self) -> int:
		return self._length

	def __iter__(self) -> Generator[dict[str, torch.Tensor], Any, None]:
		with open(self.csv_path, newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				sequence = process_sequence(row["sequence"])
				target = process_target(row["target"])

				partial, full = promptfy(sequence, target)

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

		input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
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
	is_child: bool,
) -> None:
	redis_service.set_create_status(uuid, "IN_PROGRESS")

	if is_child:
		parent_checkpoint = os.path.join(STORAGE_DIR, "models", name)
		model = AutoModelForCausalLM.from_pretrained(parent_checkpoint, low_cpu_mem_usage=False)
		tokenizer = AutoTokenizer.from_pretrained(parent_checkpoint)

	else:
		model = AutoModelForCausalLM.from_pretrained(checkpoint)
		tokenizer = AutoTokenizer.from_pretrained(checkpoint)

		special_tokens = ["[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]", "[DNA_R]", "[DNA_Y]", "[DNA_S]", "[DNA_W]", "[DNA_K]", "[DNA_M]", "[DNA_B]", "[DNA_D]", "[DNA_H]", "[DNA_V]", "[DNA_N]", "[PROT_A]", "[PROT_C]", "[PROT_D]", "[PROT_E]", "[PROT_F]", "[PROT_G]", "[PROT_H]", "[PROT_I]", "[PROT_K]", "[PROT_L]", "[PROT_M]", "[PROT_N]", "[PROT_P]", "[PROT_Q]", "[PROT_R]", "[PROT_S]", "[PROT_T]", "[PROT_V]", "[PROT_W]", "[PROT_Y]", "[PROT_*]", "[PROT_X]"]
		tokenizer.add_tokens(special_tokens)

		tokenizer.pad_token = "[PROT_*]"
		tokenizer.eos_token = "[PROT_*]"

		tokenizer.add_special_tokens({
			"eos_token": "[PROT_*]",
			"additional_special_tokens": ["<|DNA|>", "<|PROTEIN|>"]
		})

		tokenizer.add_eos_token = True

		model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

	output_path = os.path.join(STORAGE_DIR, "models", name)
	model.save_pretrained(output_path)
	tokenizer.save_pretrained(output_path)

	redis_service.set_create_status(uuid, "DONE")

def train_model(
	name: str,
	uuid: str,
	data_length: int,
	epochs: int,
	batch_size: int,
	gradient_accumulation: int,
	lr: float,
	warmup_ratio: float,
	seed: int
) -> None:
	redis_service.set_train_status(uuid, "IN_PROGRESS")

	set_seed(seed)

	accelerator = Accelerator()
	is_main_process = accelerator.is_main_process
	num_gpus = accelerator.num_processes

	if is_main_process:
		redis_service.set_train_gpu_amount = num_gpus

	checkpoint_path = os.path.join(STORAGE_DIR, "models", name)
	model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
	tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

	data_path = os.path.join(SHARED_DIR, "temp", uuid)

	dataset = DNADatasetFinetune(csv_path=data_path+".csv", tokenizer=tokenizer, dataset_total_length=data_length)
	dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=FinetuneDataCollator(tokenizer))

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
	
	redis_service.set_train_total_steps(uuid, max_train_steps)

	print("cheguei aqui")
	print("max_train_steps:", max_train_steps)

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

				redis_service.set_train_loss(uuid, loss_val)
				redis_service.set_train_lr(uuid, current_lr)
				redis_service.set_train_step(uuid, global_step)

				history["epoch"].append(round(current_epoch_fraction, 2))
				history["train_loss"].append(accumulated_loss)
				history["lr"].append(current_lr)
				history["time"].append(time.time() - start_time)

				accumulated_loss = 0.0
	
	model = accelerator.unwrap_model(model)

	if is_main_process:
		output_path = os.path.join(STORAGE_DIR, "models", name)
		model.save_pretrained(output_path)
		tokenizer.save_pretrained(output_path)

		df = pd.DataFrame(history)
		now = datetime.now().strftime("%Y%m%d-%H%M%S")
		df.to_csv(f"{output_path}/history-{now}.csv", index=False)
	
	accelerator.wait_for_everyone()
	accelerator.end_training()

	redis_service.set_train_status(uuid, "DONE")
