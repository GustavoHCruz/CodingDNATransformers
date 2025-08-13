import csv
import os
import random
import sys
import time
import tokenize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from math import ceil
from typing import Any, Generator

import torch
from accelerate import Accelerator
from config import SHARED_DIR, STORAGE_DIR
from redis_service import (CreateField, EvalField, ProcessingStatus,
                           TrainField, redis_service)
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler
from utils import load_model, save_model, set_seed


def process_sequence(sequence) -> str:
	return f"".join(f"[{nucl.upper()}]" for nucl in sequence)

def process_target(label) -> str:
	return f"[{label.upper()}]"

def promptfy(
	sequence: str,
	organism: str,
	hide_prob: float,
	target: str,
	gene: str | None,
	flank_before: str | None,
	flank_after: str | None,
) -> tuple[str, str]:
	output = f"<|SEQUENCE|>{sequence}\n"

	if organism:
		if random.random() > hide_prob:
			output += f"<|ORGANISM|>{organism[:10]}\n"

	if gene:
		if random.random() > hide_prob:
			output += f"<|GENE|>{gene[:10]}\n"
	
	if flank_before:
		if random.random() > hide_prob:
			output += f"<|FLANK_BEFORE|>{flank_before}\n"
	
	if flank_after:
		if random.random() > hide_prob:
			output += f"<|FLANK_AFTER|>{flank_after}\n"
	
	output += "<|TARGET|>"

	return output, f"{output}{target}"

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
	redis_service.set_create_info(uuid, CreateField.STATUS, ProcessingStatus.IN_PROGRESS)

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

	redis_service.set_create_info(uuid, CreateField.STATUS, ProcessingStatus.DONE)

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
	redis_service.set_train_info(uuid, TrainField.STATUS, ProcessingStatus.IN_PROGRESS)

	set_seed(seed)

	accelerator = Accelerator()
	is_main_process = accelerator.is_main_process
	num_gpus = accelerator.num_processes

	if is_main_process:
		redis_service.set_train_info(
			uuid=uuid,
			field=TrainField.GPU_AMOUNT,
			value=num_gpus
		)

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

	if is_main_process:
		redis_service.set_train_info(
			uuid=uuid,
			field=TrainField.TOTAL_STEPS,
			value=max_train_steps
		)

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
					redis_service.set_train_info(
						uuid=uuid,
						field=TrainField.LOSS,
						value=loss_val
					)
					redis_service.set_train_info(
						uuid=uuid,
						field=TrainField.LR,
						value=current_lr
					)
					redis_service.set_train_info(
						uuid=uuid,
						field=TrainField.STEP,
						value=global_step
					)

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

	redis_service.set_train_info(
		uuid=uuid,
		field=TrainField.STATUS,
		value=ProcessingStatus.DONE
	)

def evaluate(
	model_name: str,
	uuid: str,
	data_length: int,
	batch_size: int,
	seed: int
):
	redis_service.set_eval_info(uuid, EvalField.STATUS, ProcessingStatus.IN_PROGRESS)
	
	set_seed(seed)

	model, tokenizer = load_model(model_name)

	data_path = os.path.join(SHARED_DIR, "temp", uuid)

	dataset = DNADatasetFinetune(
		csv_path=data_path+".csv",
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

	model.eval()
	with torch.no_grad():
		for batch in dataloader:
			input_ids, attention_mask, labels = [b.to(model.device) for b in batch]
		
			for i, a, l in zip(input_ids, attention_mask, labels):
				prediction = model.generate(
					input_ids=i,
					attention_mask=a,
					max_new_tokens=1,
				)

				if prediction[0][-1] == labels[-1]:
					