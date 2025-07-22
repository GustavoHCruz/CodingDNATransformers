import csv
import os
import random
import time
from math import ceil
from typing import Any, Generator

import torch
from accelerate import Accelerator
from config import SHARED_DIR, STORAGE_DIR
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
			sequence_max_length: int = 1024,
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
	is_child: bool
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

def evaluate(
	model_name: str,
	uuid: str,
	data_length: int
) -> tuple[float, float, float]:
	model, tokenizer = load_model(model_name)

	return 0,0,0

def predict(
	model_name: str,
	uuid: str,
	input_text: str
) -> str:
	model, tokenizer = load_model(model_name)

	return ''

def evaluate(self):
		if not hasattr(self, "test_dataloader"):
			raise ValueError("Can't find the test dataloader, make sure you initialized it.")
		
		if not hasattr(self, "_logs_dir"):
			self._get_next_model_dir()

		self.model.to(self._device)
		total_loss = 0
		total_correct = 0
		total_samples = 0
		exon_correct = 0
		exon_total = 0
		intron_correct = 0
		intron_total = 0

		self.model.eval()
		with torch.no_grad():
			if self.log_level == "info":
				eval_bar = tqdm(self.test_dataloader, desc="Evaluating", leave=True)
			for batch in self.test_dataloader:
				input_ids, attention_mask, labels = [b.to(self._device) for b in batch]

				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
				loss = outputs.loss
				total_loss += loss.item()

				filtered_input_ids = [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask)]
				preds = []
				for filtered_input in filtered_input_ids:
					prediction = self.model.generate(
						input_ids=filtered_input.unsqueeze(0),
						attention_mask=torch.tensor([1]*filtered_input.size(-1)).unsqueeze(0).to(self._device),
						repetition_penalty=2.0,
						max_new_tokens=1,
						pad_token_id=self.tokenizer.eos_token_id
					)

					preds.append(prediction[0][filtered_input.size(-1)])

				label_texts = [label[label != -100] for label in labels]

				for pred, label in zip(preds, label_texts):
					if pred == label:
						total_correct += 1

						if label.item() == self.exon_token[0]:
							exon_correct += 1
						else:
							intron_correct += 1

					if label.item() == self.exon_token[0]:
						exon_total += 1
					else:
						intron_total += 1

					total_samples += 1

				if self.log_level == "info":
					eval_bar.update(1)
					eval_bar.set_postfix(loss=total_loss/eval_bar.n)

		if self.log_level == "info":
			eval_bar.close()		
		avg_loss = total_loss / len(self.test_dataloader)
		overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
		exon_accuracy = exon_correct / exon_total if exon_total > 0 else 0.0
		intron_accuracy = intron_correct / intron_total if intron_total > 0 else 0.0

		print(f"Evaluation complete")
		print(f"Average loss: {avg_loss:.4f}")
		print(f"Overall Accuracy: {overall_accuracy:.4f}")
		print(f"Exon accuracy: {exon_accuracy:.4f}")
		print(f"Intron accuracy: {intron_accuracy:.4f}")

		self._eval_results = {
			"avg loss": avg_loss,
			"overall accuracy": overall_accuracy,
			"exon accuracy": exon_accuracy,
			"intron accuracy": intron_accuracy
		}

		self._save_evaluation_results()

		if self.notification:
			notification.notify(title="Evaluation complete", timeout=5)
	
	def _prediction_mapping(self, prediction):
		return prediction.replace("[", "").replace("]", "").lower()

	def predict_single(self, data, map_pred=True):
		sequence = self._process_sequence(data["sequence"])
		
		keys = ["gene", "organism", "flank_before", "flank_after"]
		input_text = f"Sequence: {sequence}\n"
		for key in keys:
			if hasattr(data, key):
				input_text += f"{key.capitalize()}: {data[key]}\n"
		input_text += "Answer: "

		input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self._device)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=torch.tensor([1]*input_ids.size(-1)).unsqueeze(0).to(self._device),
				max_new_tokens=1,
				repetition_penalty=2.0,
				pad_token_id=self.tokenizer.eos_token_id,
			)

		generated_token_ids = outputs[0]
		new_token = self.tokenizer.decode(generated_token_ids[input_ids.size(-1)], skip_special_tokens=True).strip()

		if map_pred:
			return self._prediction_mapping(new_token)
		
		return new_token