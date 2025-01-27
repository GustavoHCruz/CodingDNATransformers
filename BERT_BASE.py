import json
import random

import numpy as np
import torch
from plyer import notification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

class SpliceBERTDataset(Dataset):
	def __init__(self, data, tokenizer, sequence_len, flanks_len, feat_hide_prob):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = sequence_len + flanks_len * 2 + 100
		self.feat_hide_prob = feat_hide_prob

	def __len__(self):
		return len(self.data["sequence"])
	
	def __getitem__(self, idx):
		prompt = f"Sequence:{self.data['sequence'][idx]}[SEP]"

		if len(self.data["organism"]) > idx and self.data["organism"][idx]:
			if random.random() > self.feat_hide_prob:
				prompt += f"Organism:{self.data["organism"][idx][:20]}[SEP]"
		
		if len(self.data["gene"]) > idx and self.data["gene"][idx]:
			if random.random() > self.feat_hide_prob:
				prompt += f"Gene:{self.data["gene"][idx][:20]}[SEP]"

		if len(self.data["flank_before"]) > idx and self.data["flank_before"][idx]:
			if random.random() > self.feat_hide_prob:
				prompt += f"Flank Before:{self.data["flank_before"][idx]}[SEP]"

		if len(self.data["flank_after"]) > idx and self.data["flank_after"][idx]:
			if random.random() > self.feat_hide_prob:
				prompt += f"Flank After:{self.data["flank_after"][idx]}[SEP]"
		
		prompt += "Answer:"

		input_ids = self.tokenizer.encode(prompt, max_length=self.max_length, padding=True, truncation=True)
		label = self.data["label"][idx]

		return torch.tensor(input_ids), torch.tensor(label)
class SpliceBERT():
	def _set_seed(self, seed):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def __init__(self, checkpoint="bert-base-uncased", device="cuda", seed=None, notification=False):
		"""
		A class to train and evaluate a BERT-based neural network for introns and exons classification.

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
		self._additional_info_filename = "additional_info"

		if (checkpoint != "bert-base-uncased"):
			self.load_checkpoint(checkpoint)
		else:
			self.model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
			self.tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)

		if seed is not None:
			self._set_seed(seed)

		self.notification = notification

		self.model.to(self._device)

		if checkpoint == "bert-base-uncased":
			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
	
	def load_checkpoint(self, checkpoint):
		self.model = BertForSequenceClassification.from_pretrained(checkpoint)
		self.tokenizer = BertTokenizer.from_pretrained(checkpoint)

		with open(f"{checkpoint}/{self._additional_info_filename}.json", "r") as f:
			self._last_train_info = json.load(f)
	
	def _collate_fn(self, batch):
		input_ids, labels = zip(*batch)
		input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
		return input_ids_padded, attention_mask, torch.tensor(labels)

	def _process_sequence(self, sequence):
		return f"".join(f"[{nucl.upper()}]" for nucl in sequence)
	
	def _process_label(self, label):
		return 0 if label == "intron" else 1
	
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

		self._data_configuration = {
			"sequence_len": sequence_len,
			"flanks_len": flanks_len,
			"batch_size": batch_size,
			"feat_hide_prob": feat_hide_prob
		}
		
		data = self._process_data(data)

		dataset = SpliceBERTDataset(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)

		if train_percentage == 1.0:
			self.train_dataset = dataset
		else:
			total_size = len(dataset)
			train_size = int(total_size * train_percentage)
			eval_size = total_size - train_size
			self.train_dataset, self.eval_dataset = random_split(dataset, [train_size, eval_size])
			self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def add_test_data(self, data, sequence_len=512, flanks_len=10, batch_size=32, feat_hide_prob=0.01):
		if hasattr(self, "_last_train_info"):
			if self._last_train_info["sequence_len"] != sequence_len or \
			self._last_train_info["flanks_len"] != flanks_len or \
			self._last_train_info["batch_size"] != batch_size or \
			self._last_train_info["feat_hide_prob"] != feat_hide_prob:
				print("Detected a different test dataloader configuration than the one used during training. This may lead to suboptimal results.")

		data = self._process_data(data)

		self.test_dataset = SpliceBERTDataset(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

	def free_data(self, train=True, test=True):
		if train:
			self.train_dataset = None
			self.train_dataloader = None
			self.eval_dataset = None
			self.eval_dataloader = None

		if test:	
			self.test_dataset = None
			self.test_dataloader = None

	def train(self, lr=2e-5, epochs=3, save_at_end=None, evaluation=True):
		if not hasattr(self, "train_dataloader"):
			raise ValueError("Can't find the train dataloader, make sure you initialized it.")
		
		self._last_train_info = self._data_configuration.copy()
		self._last_train_info.update({"lr": lr, "epochs": epochs})

		self.model.to(self._device)
		optimizer = AdamW(self.model.parameters(), lr=lr)

		history = {"train_loss": [], "eval_loss": [], "eval_accuracy": []}

		for epoch in range(epochs):
			self.model.train()
			train_loss = 0

			train_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=True)
			for batch in self.train_dataloader:
				optimizer.zero_grad()

				input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

				loss = outputs.loss
				loss.backward()
				optimizer.step()
				train_loss += loss.item()

				train_bar.update(1)
				train_bar.set_postfix({"Loss": train_loss/train_bar.n})

			train_loss /= len(self.train_dataloader)
			history["train_loss"].append(train_loss)
			train_bar.set_postfix({"Loss": train_loss})
			train_bar.close()

			if evaluation:
				self.model.eval()
				eval_loss = 0
				correct_predictions = 0
				total_predictions = 0

				eval_bar = tqdm(self.eval_dataloader, desc="Validating", leave=True)
				with torch.no_grad():
					for batch in self.eval_dataloader:
						input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
						outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

						loss = outputs.loss
						eval_loss += loss.item()

						predictions = torch.argmax(outputs.logits, dim=-1)
						correct_predictions += (predictions == labels).sum().item()
						total_predictions += labels.size(0)

						eval_bar.update(1)
						eval_bar.set_postfix({"Eval loss": eval_loss/eval_bar.n})

				eval_loss /= len(self.eval_dataloader)
				eval_accuracy = correct_predictions / total_predictions
				eval_bar.set_postfix({"Eval loss": eval_loss, "Eval Accuracy": eval_accuracy})
				eval_bar.close()

				history["eval_loss"].append(eval_loss)
				history["eval_accuracy"].append(eval_accuracy)

		if self.notification:
			notification.notify(title="Training complete", timeout=5)

		if save_at_end:
			self.save_checkpoint(save_at_end)

		torch.cuda.empty_cache()

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
				input_ids, labels, attention_mask = [b.to(self._device) for b in batch]

				filtered_input_ids = [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask)]

				outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
				loss = outputs.loss
				total_loss += loss.item()

				preds = []
				for filtered_input in filtered_input_ids:
					prediction = self.model.generate(
						input_ids=filtered_input.unsqueeze(0),
						attention_mask=torch.tensor([1]*filtered_input.size(-1)).unsqueeze(0).to(self._device),
						repetition_penalty=2.0,
						max_new_tokens=1,
						pad_token_id=self.tokenizer.eos_token_id
					)

					preds.append(self.tokenizer.decode(prediction[0][filtered_input.size(-1)], skip_special_tokens=True).strip())

				label_texts = [self.tokenizer.decode(label[label != -100], skip_special_tokens=True).strip() for label in labels]

				for pred, label in zip(preds, label_texts):
					if pred == label:
						total_correct += 1

						if label == "[EXON]":
							exon_correct += 1
						elif label == "[INTRON]":
							intron_correct += 1
				
					if label == "[EXON]":
						exon_total += 1
					elif label == "[INTRON]":
						intron_total += 1

					total_samples += 1

				eval_bar.update(1)
				eval_bar.set_postfix(loss=total_loss/eval_bar.n)
		
		avg_loss = total_loss / data_len
		overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
		exon_accuracy = exon_correct / exon_total if exon_total > 0 else 0.0
		intron_accuracy = intron_correct / intron_total if intron_total > 0 else 0.0

		print(f"Evaluation complete. Average loss: {avg_loss:.4f}")
		print(f"Overall Accuracy: {overall_accuracy:.4f}")
		print(f"EXON accuracy: {exon_accuracy:.4f}")
		print(f"INTRON accuracy: {intron_accuracy:.4f}")

		if self.notification:
			notification.notify(
				title="Evaluation complete",
				message=(
					f"Avg Loss: {avg_loss:.4f}, "
					f"Overall Accuracy: {overall_accuracy:.4f}, "
					f"EXON accuracy: {exon_accuracy:.4f}, "
					f"INTRON accuracy: {intron_accuracy:.4f}"
				),
				timeout=5
			)
	
	def _prediction_token_mapping(self, token):
		return "intron" if torch.argmax(token.logits, dim=-1).tolist() == [0] else "exon"

	def predict_single(self, data, map_pred=True):
		sequence = self._process_sequence(data["sequence"])
		
		keys = ["gene", "organism", "flank_before", "flank_after"]
		prompt = f"Sequence: {sequence}\n"
		for key in keys:
			if hasattr(data, key):
				prompt += f"{key.capitalize()}: {data[key]}\n"
		prompt += "Answer: "

		input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)

		self.model.eval()
		with torch.no_grad():
			prediction = self.model(input_ids=input_ids)

		if map_pred:
			return self._prediction_token_mapping(prediction)
		
		return prediction
	
	def predict_batch(self, data_batch, map_pred=True):
		preds = []
		for data in data_batch:
			pred = self.predict_single(data, map_pred)
			preds.append(pred)
		
		return preds
		
	def save_checkpoint(self, checkpoint):
		if not hasattr(self, "_last_train_info"):
			raise ValueError("Nothing to save")
		
		self.model.save_pretrained(checkpoint)
		self.tokenizer.save_pretrained(checkpoint)

		with open(f"{checkpoint}/{self._additional_info_filename}.json", "w") as f:
			json.dump(self._last_train_info, f, indent=2)
		
		print(f"Model & Infos Successful Saved at {checkpoint}")

import pandas as pd

df = pd.read_csv("datasets/ExInSeqs_100k_small.csv", keep_default_na=False)

sequence = df.iloc[:, 0].tolist()
label = df.iloc[:, 1].tolist()
organism = df.iloc[:, 2].tolist()
gene = df.iloc[:, 3].tolist()
flank_before = df.iloc[:, 4].tolist()
flank_after = df.iloc[:, 5].tolist()

splicebert = SpliceBERT()

splicebert.add_train_data({
	"sequence": sequence,
	"label": label,
	"organism": organism,
	"gene": gene,
	"flank_before": flank_before,
	"flank_after": flank_after
}, sequence_len=256, flanks_len=10, batch_size=8, feat_hide_prob=0.2)

splicebert.train(save_at_end="models/splicebert")