import random
import time

import torch
from plyer import notification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from SplicingTransformers import SplicingTransformers

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

class SpliceGPT(SplicingTransformers):
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

	def __init__(self, checkpoint="gpt2", device="cuda", seed=None, notification=False, logs_dir="logs", models_dir="models", alias=None):
		if seed:
			self._set_seed(seed)
			
		supported = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "bigscience/bloom-560m"]
		
		if checkpoint not in supported:
			self.load_checkpoint(checkpoint)
		else:
			self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
			self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
			
			self.tokenizer.pad_token = self.tokenizer.eos_token

			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[EXON]", "[INTRON]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer))

		self.intron_token = self.tokenizer.encode("[INTRON]", add_special_tokens=False)
		self.exon_token = self.tokenizer.encode("[EXON]", add_special_tokens=False)

		super().__init__(checkpoint=checkpoint, device=device, seed=seed, notification=notification, logs_dir=logs_dir, models_dir=models_dir, alias=alias)

	def load_checkpoint(self, path):
		self.model = AutoModelForCausalLM.from_pretrained(path)
		self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")

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

	def add_train_data(self, data, batch_size=32, sequence_len=512, train_percentage=0.8, data_config=None):
		flanks_len = 10
		feat_hide_prob = 0.01
		if hasattr(data_config, "flanks_len"):
			flanks_len = data_config["flanks_len"]
		if hasattr(data_config, "feat_hide_prob"):
			feat_hide_prob = data_config["feat_hide_prob"]

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

	def _check_test_compatibility(self, sequence_len, flanks_len, batch_size, feat_hide_prob):
		if hasattr(self, "_train_config"):
			if self._train_config["sequence_len"] != sequence_len or \
			self._train_config["flanks_len"] != flanks_len or \
			self._train_config["batch_size"] != batch_size or \
			self._train_config["feat_hide_prob"] != feat_hide_prob:
				print("Detected a different test dataloader configuration of the one used during training. This may lead to suboptimal results.")

	def add_test_data(self, data, batch_size=32, sequence_len=512, data_config=None):
		flanks_len = 10
		feat_hide_prob = 0.01
		if hasattr(data_config, "flanks_len"):
			flanks_len = data_config["flanks_len"]
		if hasattr(data_config, "feat_hide_prob"):
			feat_hide_prob = data_config["feat_hide_prob"]

		self._check_test_compatibility(sequence_len, flanks_len, batch_size, feat_hide_prob)
		data = self._process_data(data)

		self.test_dataset = self.__SpliceGPTDataset__(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def train(self, lr=5e-4, epochs=3, evaluation=True, save_at_end=True, keep_best=False, save_freq=5):
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
		if evaluation:
			history.update({"eval_loss": []})

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
				best = False
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

			if evaluation and eval_loss < best_eval_loss:
				best = True
				best_eval_loss = eval_loss
				self._save_checkpoint()
			
			self.epoch_end_time = time.time()
			history["time"].append(self.epoch_end_time - self.start_time)
		
		if keep_best:
			if evaluation:
				if not best:
					self._load_checkpoint()
			else:
				print("Cannot load best because evaluation is setted off. To be able to restore the best model from the stored ones, please allow evaluation to execute with trainning.")

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

				eval_bar.update(1)
				eval_bar.set_postfix(loss=total_loss/eval_bar.n)

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

		self._seve_evaluation_results()

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
