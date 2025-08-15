import time

import torch
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
		def __init__(self, data, tokenizer, window_size, flank_size):
			self.data = data
			self.tokenizer = tokenizer
			self.max_length = window_size + flank_size * 2 + 75

		def __len__(self):
			return len(self.data)
		
		def __getitem__(self, idx):
			prompt = f"<|SEQUENCE|>{self.data[idx]["sequence"]}[SEP]"
			prompt = f"<|FLANK_BEFORE|>{self.data[idx]["before"]}[SEP]"
			prompt = f"<|FLANK_AFTER|>{self.data[idx]["after"]}[SEP]"
			prompt += f"<|ORGANISM|>{self.data[idx]["organism"][:10]}[SEP]"
			
			prompt += "<|TARGET|>"

			input_ids = self.tokenizer.encode(prompt, max_length=self.max_length, padding=True, truncation=True)
			label = self.data[idx]["label"]

			return torch.tensor(input_ids), torch.tensor(label)

	def __init__(self, checkpoint="bert-base-uncased"):
		label_to_index, index_to_label = create_label_mapping(self.window_size)
		self.label_to_index = label_to_index
		self.index_to_label = index_to_label

		self.model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(self.label_to_index))
		self.tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)
	
		special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[I]", "[E]", "[U]"]
		self.tokenizer.add_tokens(special_tokens)

		self.tokenizer.add_special_tokens({
			"additional_special_tokens": ["<|SEQUENCE|>", "<|ORGANISM|>", "<|GENE|>", "<|FLANK_BEFORE|>", "<|FLANK_AFTER|>", "<|TARGET|>"]
		})


		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

	def _collate_fn(self, batch):
		input_ids, labels = zip(*batch)
		input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
		return input_ids_padded, attention_mask, torch.tensor(labels)
	
	def _process_sequence(self, sequence):
		return f"".join(f"[{nucl.upper()}]" for nucl in sequence)
	
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
		
		for sequence, organism, labeled_sequence in zip(*data.values()):
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

				final_data.append({
					"sequence": seq,
					"before": flank_before,
					"after": flank_after,
					"label": label,
					"organism": organism
				})

		return final_data
	
	def add_train_data(self, data):
		data = self._process_data(data)

		dataset = self.__SWExInBERT__(data, self.tokenizer, self.window_size, self.flank_size)

		self.train_dataset = dataset
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=self._collate_fn)

	def add_test_data(self, data):
		data = self._process_data(data)

		self.test_dataset = self.__SWExInBERT__(data, self.tokenizer, window_size=self.window_size, flank_size=self.flank_size)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=True, collate_fn=self._collate_fn)

	def train(self, lr=2e-5, epochs=1):
		if not hasattr(self, "train_dataloader"):
			raise ValueError("Cannot find the train dataloader, make sure you initialized it.")
		
		self.start_time = time.time()
		
		self.model.to("cuda")
		self.optimizer = AdamW(self.model.parameters(), lr=lr)

		history = {"epoch": [], "time": [], "train_loss": []}

		for epoch in range(epochs):
			self.model.train()
			train_loss = 0

			train_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=True)
			for batch in self.train_dataloader:
				self.optimizer.zero_grad()

				input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

				loss = outputs.loss
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()

				train_bar.update(1)
				train_bar.set_postfix({"Loss": train_loss/train_bar.n})

			train_loss /= len(self.train_dataloader)
			history["train_loss"].append(train_loss)
			train_bar.set_postfix({"Loss": train_loss})
			train_bar.close()

			self.epoch_end_time = time.time()
			history["time"].append(self.epoch_end_time - self.start_time)
			history["epoch"].append(epoch)

		torch.cuda.empty_cache()

		self.model.save_pretrained("./modelao")
		self.tokenizer.save_pretrained("./modelao")

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
