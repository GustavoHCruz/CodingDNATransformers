import time

import torch
from plyer import notification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

from classes.SplicingTransformers import SplicingTransformers

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

def create_label_mapping(window_size):
	from itertools import product
	
	possible_labels = ['I', 'E', 'U']
	label_combinations = [''.join(p) for p in product(possible_labels, repeat=window_size)]
	
	label_to_index = {label: idx for idx, label in enumerate(label_combinations)}
	index_to_label = {idx: label for label, idx in label_to_index.items()}
	
	return label_to_index, index_to_label

class SWExInSeqsBERT(SplicingTransformers):
	class __SWExInBERT__(Dataset):
		def __init__(self, data, tokenizer, window_size, flank_size):
			self.data = data
			self.tokenizer = tokenizer
			self.max_length = window_size + flank_size * 2 + 75

		def __len__(self):
			return len(self.data)
		
		def __getitem__(self, idx):
			prompt = f"Sequence:{self.data[idx]["sequence"]}[SEP]"
			prompt = f"Flank Before: {self.data[idx]["before"]}[SEP]"
			prompt = f"Flank After: {self.data[idx]["after"]}[SEP]"
			prompt += f"Organism:{self.data[idx]["organism"][:20]}[SEP]"
			
			prompt += "Answer:"

			input_ids = self.tokenizer.encode(prompt, max_length=self.max_length, padding=True, truncation=True)
			label = self.data[idx]["label"]

			return torch.tensor(input_ids), torch.tensor(label)

	def __init__(self, checkpoint="bert-base-uncased", device="cuda", seed=None, notification=False,  logs_dir="logs", models_dir="models", alias=None, log_level="info", window_size=3):
		if seed:
			self._set_seed(seed)

		self.log_level = log_level

		self.window_size = window_size
		
		label_to_index, index_to_label = create_label_mapping(window_size)
		self.label_to_index = label_to_index
		self.index_to_label = index_to_label

		if checkpoint != "bert-base-uncased":
			self.load_checkpoint(checkpoint)
		else:
			self.model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(self.label_to_index))
			self.tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)
		
		if checkpoint == "bert-base-uncased":
			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[I]", "[E]", "[U]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

		super().__init__(checkpoint=checkpoint, device=device, seed=seed, notification=notification, logs_dir=logs_dir, models_dir=models_dir, alias=alias)
		
	def load_checkpoint(self, path):
		self.model = BertForSequenceClassification.from_pretrained(path)
		self.tokenizer = BertTokenizer.from_pretrained(path)

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
	
	def _process_data(self, data, flank_size):
		window_size = self.window_size
		final_data = []
		
		for sequence, organism, labeled_sequence in zip(*data.values()):
			seq_length = len(sequence)
			
			for i in range(seq_length - window_size + 1):
				flank_start = max(i - flank_size, 0)
				flank_end = min(i + window_size + flank_size, seq_length)
				
				seq = sequence[i:i + window_size]
				flank_before = sequence[flank_start:i]
				flank_after = sequence[i + window_size:flank_end]
				label = labeled_sequence[i:i + window_size]
				
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
	
	def add_train_data(self, data, batch_size=32, data_config=None):
		flank_size = 64
		if "flank_size" in data_config:
			flank_size = data_config["flank_size"]

		self._data_config = {
			"window_size": self.window_size,
			"flank_size": flank_size,
			"batch_size": batch_size,
		}
		
		data = self._process_data(data, flank_size=flank_size)

		dataset = self.__SWExInBERT__(data, self.tokenizer, self.window_size, flank_size)

		self.train_dataset = dataset
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def _check_test_compatibility(self, flank_size, batch_size):
		if hasattr(self, "_train_config"):
			if self._train_config["flanks_size"] != flank_size or self._train_config["batch_size"] != batch_size:
				print("Detected a different test dataloader configuration of the one used during training. This may lead to suboptimal results.")

	def add_test_data(self, data, batch_size=32, data_config=None):
		flank_size = 64
		if "flank_size" in data_config:
			flank_size = data_config["flank_size"]

		self._check_test_compatibility(flank_size=flank_size, batch_size=batch_size)

		data = self._process_data(data, flank_size=flank_size)

		self.test_dataset = self.__SWExInBERT__(data, self.tokenizer, window_size=self.window_size, flank_size=flank_size)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def train(self, lr=2e-5, epochs=3, save_at_end=None, save_freq=5):
		if not hasattr(self, "train_dataloader"):
			raise ValueError("Cannot find the train dataloader, make sure you initialized it.")
		
		self.start_time = time.time()
		self._get_next_model_dir()
		
		self.model.to(self._device)
		self.optimizer = AdamW(self.model.parameters(), lr=lr)

		self._train_config = dict(**{
			"lr": lr,
			"epochs": epochs
		}, **self._data_config)
		if hasattr(self, "seed"):
			self._train_config.update({
				"seed": self.seed
			})

		history = {"epoch": [], "time": [], "train_loss": []}

		for epoch in range(epochs):
			self.model.train()
			train_loss = 0

			if self.log_level == "info":
				train_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=True)
			for batch in self.train_dataloader:
				self.optimizer.zero_grad()

				input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

				loss = outputs.loss
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()

				if self.log_level == "info":
					train_bar.update(1)
					train_bar.set_postfix({"Loss": train_loss/train_bar.n})

			train_loss /= len(self.train_dataloader)
			# history["train_loss"].append(train_loss)
			if self.log_level == "info":
				train_bar.set_postfix({"Loss": train_loss})
				train_bar.close()

			#if save_freq and (epoch+1) % save_freq == 0:
				#self._save_checkpoint(epoch=epoch)

		if self.notification:
			notification.notify(title="Training complete", timeout=5)

		torch.cuda.empty_cache()

		#self._save_history(history=history)
		#self._save_config()

		if save_at_end:
			self.save_checkpoint()

	def evaluate(self):
		if not hasattr(self, "test_dataloader"):
			raise ValueError("Can't find the test dataloader, make sure you initialized it.")
		
		if not hasattr(self, "_logs_dir"):
			self._get_next_model_dir()

		self.model.to(self._device)
		
		self.model.eval()
		total_loss = 0
		total_correct = 0
		total_samples = 0

		with torch.no_grad():
			if self.log_level == "info":
				eval_bar = tqdm(self.test_dataloader, desc="Evaluating", leave=True)
			for batch in self.test_dataloader:
				input_ids, attention_mask, label_ids = [b.to(self._device) for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)

				loss = outputs.loss
				total_loss += loss.item()

				predictions = torch.argmax(outputs.logits, dim=-1)

				for prediction, label in zip(predictions, label_ids):
					if prediction == label:
						total_correct += 1
					total_samples += 1
				
				if self.log_level == "info":
					eval_bar.update(1)
					eval_bar.set_postfix({"Eval loss": total_loss/eval_bar.n})
			
			if self.log_level == "info":
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

		self._save_evaluation_results()

		if self.notification:
			notification.notify(title="Evaluation complete", timeout=5)
	
	def _prediction_mapping(self, prediction):
		return self.label_to_index[prediction]
		
	def predict_single(self, data, map_pred=True):
		if map_pred:
			return self._prediction_mapping()
		
		return 