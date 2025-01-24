import json
import os
import random

import numpy as np
import pandas as pd
import torch
from plyer import notification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

class SpliceGPT():
	class __SpliceGPTDataset__(Dataset):
		def __init__(self, data, tokenizer, sequence_len, flanks_len, feat_hide_prob):
			self.data = data
			self.tokenizer = tokenizer
			self.max_length = sequence_len + flanks_len * 2 + 80
			self.feat_hide_prob = feat_hide_prob

		def __len__(self):
			return len(self.data["sequence"])
		
		def __getitem__(self, idx):
			input_text = f"Sequence:{self.data['sequence'][idx]}\n"

			if len(self.data["organism"]) > idx and self.data["organism"][idx]:
				if random.random() > self.feat_hide_prob:
					input_text += f"Organism:{self.data["organism"][idx][:10]}\n"
			
			if len(self.data["gene"]) > idx and self.data["gene"][idx]:
				if random.random() > self.feat_hide_prob:
					input_text += f"Gene:{self.data["gene"][idx][:10]}\n"

			if len(self.data["flank_before"]) > idx and self.data["flank_before"][idx]:
				if random.random() > self.feat_hide_prob:
					input_text += f"Flank Before:{self.data["flank_before"][idx]}\n"

			if len(self.data["flank_after"]) > idx and self.data["flank_after"][idx]:
				if random.random() > self.feat_hide_prob:
					input_text += f"Flank After:{self.data["flank_after"][idx]}\n"
			
			input_text += "Answer:"
			output_text = f"{self.data["label"][idx]}"

			input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length, add_special_tokens=True, padding=True)
			label_ids = self.tokenizer.encode(output_text, truncation=True, max_length=self.max_length, add_special_tokens=False)

			labels = [-100] * len(input_ids)
			labels[-len(label_ids):] = label_ids

			return torch.tensor(input_ids), torch.tensor(labels)

	def _set_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def __init__(self, checkpoint="gpt2", device="cuda", seed=None, notification=False, logs_dir="logs", alias=None):
		"""
		A class to train and evaluate a GPT-based neural network for introns and exons classification.

		This class manages preprocessing, training, evaluation, and sequence classification tasks for the model. 
		It includes methods for training, creating dataloaders, evaluation, and prediction, and supports execution 
		on both GPU and CPU.

		Attributes:
			checkpoint (str): The checkpoint to start training or evaluation from.
			device (str): The device to execute the model on ('cpu' or 'cuda').
			seed (int): A custom seed to enable deterministic results.
			notification (bool): If enabled, sends a GUI notification when training or evaluation concludes.
		"""
		self._device = device
		self.logs_dir = logs_dir
		self.checkpoint = checkpoint
		self.alias = alias or checkpoint

		if (checkpoint != "gpt2"):
			self.load_checkpoint(checkpoint)
		else:
			self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
			self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side="left")

		if seed is not None:
			self._set_seed(seed)

		self.notification = notification

		self.model.to(self._device)

		self.intron_token = self.tokenizer.encode("[INTRON]", add_special_tokens=False)
		self.exon_token = self.tokenizer.encode("[EXON]", add_special_tokens=False)

		if checkpoint == "gpt2":
			self.tokenizer.pad_token = self.tokenizer.eos_token

			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[EXON]", "[INTRON]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
	
	def _get_next_model_dir(self):
		os.makedirs(self.logs_dir, exist_ok=True)

		model_dir = os.path.join(self.logs_dir, self.alias)

		counter = 1
		while os.path.exists(f"{model_dir}_{counter}"):
			counter += 1

		model_dir = f"{model_dir}_{counter}"
		
		os.makedirs(f"{model_dir}/checkpoints")

		self._model_dir = model_dir
	
	def save_checkpoint(self, path=None):
		saving_path = f"models/{self.alias}"
		if path:
			saving_path = path

		self.model.save_pretrained(saving_path)
		self.tokenizer.save_pretrained(saving_path)

		print(f"Model Successful Saved at {saving_path}")

	def load_checkpoint(self, checkpoint):
		self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
		self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side="left")

	def _collate_fn(self, batch):
		input_ids, labels = zip(*batch)
		input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
		attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
		return input_ids_padded, attention_mask, labels_padded
	
	def _process_sequence(self, sequence):
		return f"".join(f"[{nucl.upper()}]" for nucl in sequence)
	
	def _process_label(self, label):
		return f"[{label.upper()}]"
	
	def _process_data(self, data):
		data["sequence"] = [self._process_sequence(sequence) for sequence in data["sequence"]]
		data["label"] = [self._process_label(label) for label in data["label"]]
		data["flank_before"] = [self._process_sequence(sequence) for sequence in data["flank_before"]]
		data["flank_after"] = [self._process_sequence(sequence) for sequence in data["flank_after"]]

		return data

	def add_train_data(self, data, sequence_len=512, flanks_len=10, batch_size=32, train_percentage=0.8, feat_hide_prob=0.01):
		if sequence_len > 512:
			raise ValueError("cannot support sequences_len higher than 512")
		if flanks_len > 50:
			raise ValueError("cannot support flanks_len higher than 50")

		self._data_config = {
			"sequence_len": sequence_len,
			"flanks_len": flanks_len,
			"batch_size": batch_size,
			"feat_hide_prob": feat_hide_prob
		}
		
		data = self._process_data(data)

		dataset = self.__SpliceGPTDataset__(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)

		if train_percentage == 1.0:
			self.train_dataset = dataset
		else:
			total_size = len(dataset)
			train_size = int(total_size * train_percentage)
			eval_size = total_size - train_size
			self.train_dataset, self.eval_dataset = random_split(dataset, [train_size, eval_size])
			self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def _check_test_compability(self, sequence_len, flanks_len, batch_size, feat_hide_prob):
		if hasattr(self, "_train_config"):
			if self._train_config["sequence_len"] != sequence_len or \
			self._train_config["flanks_len"] != flanks_len or \
			self._train_config["batch_size"] != batch_size or \
			self._train_config["feat_hide_prob"] != feat_hide_prob:
				print("Detected a different test dataloader configuration of the one used during training. This may lead to suboptimal results.")

	def add_test_data(self, data, sequence_len=512, flanks_len=10, batch_size=32, feat_hide_prob=0.01):
		self._check_test_compability(sequence_len, flanks_len, batch_size, feat_hide_prob)
		data = self._process_data(data)

		self.test_dataset = self.__SpliceGPTDataset__(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def free_data(self, train=True, test=True):
		if train:
			self.train_dataset = None
			self.train_dataloader = None
			self.eval_dataset = None
			self.eval_dataloader = None

		if test:	
			self.test_dataset = None
			self.test_dataloader = None

	def update_alias(self, alias):
		self.alias = alias
		self._model_dir = self._get_next_model_dir()

	def _save_checkpoint(self, epoch=None):
		if epoch != None:
			path = f"{self._model_dir}/checkpoints/checkpoint-epoch-{epoch}.pth"
		else:
			path = f"{self._model_dir}/best.pth"

		torch.save({
			"model_state_dict": self.model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
		}, path)

	def _load_checkpoint(self, epoch=None):
		if epoch != None:
			path = f"{self._model_dir}/checkpoints/checkpoint-epoch-{epoch}.pth"
		else:
			path = f"{self._model_dir}/best.pth"

		model, optimizer = torch.load(path, weights_only=True).values()
		
		self.model.load_state_dict(model)
		self.optimizer.load_state_dict(optimizer)

	def _save_history(self, history):
		df = pd.DataFrame(history)
		df.to_csv(f"{self._model_dir}/history.csv", index=False)
	
	def _save_config(self):
		with open(f"{self._model_dir}/config.json", "w") as f:
			json.dump(self._train_config, f, indent=2)

	def train(self, lr=0.0005, epochs=3, save_at_end=True, evaluation=True, keep_best=False, save_freq=5):
		if not hasattr(self, "train_dataloader"):
			raise ValueError("Can't find the train dataloader, make sure you initialized it.")
		
		self._get_next_model_dir()

		self.model.to(self._device)
		self.optimizer = AdamW(self.model.parameters(), lr=lr)

		self._train_config = dict(**{
			"lr": lr,
			"epochs": epochs
		}, **self._data_config)

		history = {"epoch": [], "train_loss": [], "eval_loss": []}

		best_eval_loss = float("inf")

		for epoch in range(epochs):
			self.model.train()
			train_loss = 0
			
			train_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=True)
			for batch in self.train_dataloader:
				self.optimizer.zero_grad()

				input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
				
				loss = outputs.loss
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()

				train_bar.update(1)
				train_bar.set_postfix(loss=train_loss/train_bar.n)
	
			train_loss /= len(self.train_dataloader)
			history["train_loss"].append(train_loss)
			train_bar.set_postfix({"Loss": train_loss})
			train_bar.close()

			if evaluation:
				self.model.eval()
				eval_loss = 0

				eval_bar = tqdm(self.eval_dataloader, desc="Validating", leave=True)
				with torch.no_grad():
					for batch in self.eval_dataloader:
						input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
						outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

						loss = outputs.loss
						eval_loss += loss.item()

						eval_bar.update(1)
						eval_bar.set_postfix({"Eval loss": eval_loss/eval_bar.n})

				eval_loss /= len(self.eval_dataloader)
				eval_bar.set_postfix({"Eval loss": eval_loss})
				eval_bar.close()
				history["eval_loss"].append(eval_loss)

			history["epoch"].append(epoch)

			if (epoch+1) % save_freq == 0:
				self._save_checkpoint(epoch=epoch)

			if eval_loss < best_eval_loss:
				best_eval_loss = eval_loss
				self._save_checkpoint()
		
		if keep_best:
			self._load_checkpoint()

		if self.notification:
			notification.notify(title="Training complete", timeout=5)

		torch.cuda.empty_cache()

		self._save_history(history=history)
		self._save_config()

		if save_at_end:
			self.save_checkpoint()
	
	def evaluate(self):
		if not hasattr(self, "test_dataloader"):
			raise ValueError("Can't find the test dataloader, make sure you initialized it.")

		self.model.to(self._device)
		total_loss = 0
		total_correct = 0
		total_samples = 0
		exon_correct = 0
		exon_total = 0
		intron_correct = 0
		intron_total = 0
		data_len = len(self.test_dataloader)

		self.model.eval()
		with torch.no_grad():
			eval_bar = tqdm(total=data_len, desc="Evaluating", position=0, leave=True)
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

				eval_bar.update(1)
				eval_bar.set_postfix(loss=total_loss/eval_bar.n)

		eval_bar.close()		
		avg_loss = total_loss / data_len
		overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
		exon_accuracy = exon_correct / exon_total if exon_total > 0 else 0.0
		intron_accuracy = intron_correct / intron_total if intron_total > 0 else 0.0

		print(f"Evaluation complete. Average loss: {avg_loss:.4f}")
		print(f"Overall Accuracy: {overall_accuracy:.4f}")
		print(f"Exon accuracy: {exon_accuracy:.4f}")
		print(f"Intron accuracy: {intron_accuracy:.4f}")

		if self.notification:
			notification.notify(title="Evaluation complete", timeout=5)
	
	def _prediction_token_mapping(self, token):
		return token.replace("[", "").replace("]", "").lower()

	def predict_single(self, data, repetition_penalty=2.0, map_pred=True):
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
				repetition_penalty=repetition_penalty,
				pad_token_id=self.tokenizer.eos_token_id,
			)

		generated_token_ids = outputs[0]
		new_token = self.tokenizer.decode(generated_token_ids[input_ids.size(-1)], skip_special_tokens=True).strip()

		if map_pred:
			return self._prediction_token_mapping(new_token)
		
		return new_token
	
	def predict_batch(self, data_batch, repetition_penalty=2.0, map_pred=True):
		preds = []
		for data in data_batch:
			pred = self.predict_single(data, repetition_penalty, map_pred)
			preds.append(pred)
		
		return preds

splicegpt = SpliceGPT(checkpoint="models/SpliceGPT", device="cuda", seed=1234, notification=True, logs_dir="logs", alias="SpliceGPT")

df = pd.read_csv("datasets/ExInSeqs_3k_small.csv", keep_default_na=False)

test_sequence = df.iloc[:1000, 0].tolist()
test_label = df.iloc[:1000, 1].tolist()
test_organism = df.iloc[:1000, 2].tolist()
test_gene = df.iloc[:1000, 3].tolist()
test_flank_before = df.iloc[:1000, 4].tolist()
test_flank_after = df.iloc[:1000, 5].tolist()

splicegpt.add_test_data({
	"sequence": test_sequence,
	"label": test_label,
  "organism": test_organism,
  "gene": test_gene,
  "flank_before": test_flank_before,
  "flank_after": test_flank_after
}, sequence_len=256, flanks_len=10, batch_size=16, feat_hide_prob=0.8)

splicegpt.evaluate()

for i in range(10):
	prediction = splicegpt.predict_single({
		"sequence": test_sequence[i],
		"organism": test_organism[i],
		"gene": test_gene[i],
		"flank_before": test_flank_before[i],
		"flank_after": test_flank_after[i]
	}, map_pred=True)

	print(f"Expected: {test_label[i]} - Prediction: {prediction}")
