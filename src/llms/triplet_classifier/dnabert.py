import time

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


def create_label_mapping(window_size):
	from itertools import product
	
	possible_labels = ['I', 'E', 'U']
	label_combinations = [''.join(p) for p in product(possible_labels, repeat=window_size)]
	
	label_to_index = {label: idx for idx, label in enumerate(label_combinations)}
	index_to_label = {idx: label for label, idx in label_to_index.items()}
	
	return label_to_index, index_to_label

class SWExInSeqsBERT():
	window_size = 3
	flank_size = 64

	class __SWExInBERT__(Dataset):
		def __init__(self, data, tokenizer):
			self.data = data
			self.tokenizer = tokenizer
			self.max_length = 512

		def __len__(self):
			return len(self.data)
		
		def __getitem__(self, idx):
			input_ids = self.tokenizer.encode(self.data[idx]["sequence"], max_length=self.max_length, padding=True, truncation=True)
			label = self.data[idx]["target"]

			return torch.tensor(input_ids), torch.tensor(label)

	def __init__(self, checkpoint="zhihan1996/DNA_bert_6"):
		label_to_index, index_to_label = create_label_mapping(self.window_size)
		self.label_to_index = label_to_index
		self.index_to_label = index_to_label

		self.model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(self.label_to_index))
		self.tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)
	
		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

	def _collate_fn(self, batch):
		input_ids, labels = zip(*batch)
		input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
		return input_ids_padded, attention_mask, torch.tensor(labels)
	
	def _process_sequence(self, sequence):
		res = sequence.replace("U", "N")
		return f"".join(f"[{res.upper()}]" for nucl in sequence)
	
	def _process_target(self, label):
		seq_length = len(label)
		processed_labels = []
		
		for i in range(seq_length - self.window_size + 1):
			label_window = label[i:i + self.window_size]
			label_idx = self.label_to_index[label_window]
			processed_labels.append(label_idx)
	
		return processed_labels
	
	def _process_data(self, data):
		final_data = []
		
		for sequence, labeled_sequence in zip(*data.values()):
			seq_length = len(sequence)
			
			for i in range(seq_length - self.window_size + 1):
				flank_start = max(i - self.flank_size, 0)
				flank_end = min(i + self.window_size + self.flank_size, seq_length)
				
				seq = sequence[i:i + self.window_size]
				flank_before = sequence[flank_start:i]
				flank_after = sequence[i + self.window_size:flank_end]
				label = labeled_sequence[i:i + self.window_size]
				
				seq = self._process_sequence(seq)
				flank_before = self._process_sequence(flank_before)
				flank_after = self._process_sequence(flank_after)

				label = self._process_target(label)
				
				final_sequence = flank_before+flank_after+seq
	
				remain = len(final_sequence) % 6
				if remain != 0:
					padding_len = 6 - remain
					final_sequence += 'N' * padding_len

				final_data.append({
					"sequence":final_sequence,
					"target": label
				})

		return final_data
	
	def add_train_data(self, data):
		data = self._process_data(data)

		dataset = self.__SWExInBERT__(data, self.tokenizer)

		self.train_dataset = dataset
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=self._collate_fn)

	def add_test_data(self, data):
		data = self._process_data(data)

		self.test_dataset = self.__SWExInBERT__(data, self.tokenizer)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=True, collate_fn=self._collate_fn)

	def train(self, lr=2e-5, epochs=1):
		if not hasattr(self, "train_dataloader"):
			raise ValueError("Cannot find the train dataloader, make sure you initialized it.")
		
		accelerator = Accelerator()
		self.start_time = time.time()
		
		self.model.to("cuda")
		self.optimizer = AdamW(self.model.parameters(), lr=lr)

		self.model, self.optimizer, self.train_dataloader = accelerator.prepare(
			self.model, self.optimizer, self.train_dataloader
		)

		history = {"epoch": [], "time": [], "train_loss": []}

		for epoch in range(epochs):
			self.model.train()
			train_loss = 0

			train_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=True,  disable=not accelerator.is_local_main_process)
			for batch in self.train_dataloader:
				self.optimizer.zero_grad()

				input_ids, attention_mask, labels = batch
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

				loss = outputs.loss
				accelerator.backward(loss)
				self.optimizer.step()
				train_loss += loss.item()

				train_bar.update(1)
				train_bar.set_postfix({"Loss": train_loss/(train_bar.n | 1)})

			train_loss /= len(self.train_dataloader)
			history["train_loss"].append(train_loss)
			train_bar.set_postfix({"Loss": train_loss})
			train_bar.close()

			self.epoch_end_time = time.time()
			history["time"].append(self.epoch_end_time - self.start_time)
			history["epoch"].append(epoch)

		torch.cuda.empty_cache()

		if accelerator.is_local_main_process:
			accelerator.unwrap_model(self.model).save_pretrained("./modelao-dna-3")
			self.tokenizer.save_pretrained("./modelao-dna-3")

	def evaluate(self):
		self.model.to("cuda")
		
		self.model.eval()
		total_loss = 0
		total_correct = 0
		total_samples = 0

		with torch.no_grad():
			eval_bar = tqdm(self.test_dataloader, desc="Evaluating", leave=True)
			for batch in self.test_dataloader:
				input_ids, attention_mask, label_ids = [b.to("cuda") for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)

				loss = outputs.loss
				total_loss += loss.item()

				predictions = torch.argmax(outputs.logits, dim=-1)

				for prediction, label in zip(predictions, label_ids):
					if prediction == label:
						total_correct += 1
					total_samples += 1
				
				eval_bar.update(1)
				eval_bar.set_postfix({"Eval loss": total_loss/eval_bar.n})
			
			eval_bar.close()
			total_loss /= len(self.test_dataloader)
			overall_accuracy = total_correct / total_samples

			print(f"Evaluation complete")
			print(f"Avarage loss: {total_loss:.4f}")
			print(f"Overall Accuracy: {overall_accuracy:.4f}")

			self._eval_results = {
				"avg loss": total_loss,
				"overall accuracy": overall_accuracy,
			}

import numpy as np
import pandas as pd

df = pd.read_csv("TRIPLET.csv")

df = df.replace({np.nan: None})

sequence = df["sequence"].tolist()
target = df["target"].tolist()
organism = df["organism"].tolist()
before = df["flankBefore"].tolist()
after = df["flankAfter"].tolist()
gene = df["gene"].tolist()

bert = SWExInSeqsBERT()

print(len(sequence))

bert.add_train_data({
  "sequence": sequence[3950:4000],
  "target": target[3950:4000]
})

bert.add_test_data({
  "sequence": sequence[5000:5030],
  "target": target[5000:5030]
})

bert.train(epochs=3)

bert.evaluate()