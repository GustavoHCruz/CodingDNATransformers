import re
import time

import torch
from plyer import notification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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

class RebuildSeqsGPT(SplicingTransformers):
	class __RebuildSeqsGPTDataset__(Dataset):
		def __init__(self, data, tokenizer):
			self.data = data
			self.tokenizer = tokenizer
			
			biggest_sequence = max(data["builded"])
			tokenized = tokenizer.encode(biggest_sequence)

			max_len = len(tokenized)
			if max_len > 1024:
				max_len = 1024

			self.max_length = max_len

		def __len__(self):
			return len(self.data["sequence"])
		
		def __getitem__(self, idx):
			input_text = f"Sequence:{self.data["sequence"][idx]}\n"
			input_text += f"Organism:{self.data["organism"][idx][:10]}\n"
			input_text += "Marked Sequence:"

			output_text = f"{self.data["builded"][idx]}"

			input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length, add_special_tokens=True, padding=True)
			target_ids = self.tokenizer.encode(output_text, truncation=True, max_length=self.max_length, add_special_tokens=True, padding=True)

			return torch.tensor(input_ids), torch.tensor(target_ids)

	def __init__(self, checkpoint="gpt2", device="cuda", seed=None, notification=False, logs_dir="logs", models_dir="models", alias=None, log_level="info"):
		if seed:
			self._set_seed(seed)
		
		self.log_level = log_level
			
		supported = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "bigscience/bloom-560m"]

		if checkpoint not in supported:
			self.load_checkpoint(checkpoint)
		else:
			self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
			self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
			
			self.tokenizer.pad_token = self.tokenizer.eos_token

			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[EXON]", "[INTRON]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

		self.intron_token = self.tokenizer.encode("[INTRON]", add_special_tokens=False)
		self.exon_token = self.tokenizer.encode("[EXON]", add_special_tokens=False)

		super().__init__(checkpoint=checkpoint, device=device, seed=seed, notification=notification, logs_dir=logs_dir, models_dir=models_dir, alias=alias)

	def load_checkpoint(self, path):
		self.model = AutoModelForCausalLM.from_pretrained(path)
		self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")

	def _collate_fn(self, batch):
		input_ids, target_ids = zip(*batch)
		
		max_len = max(max(t.shape[0] for t in input_ids), max(t.shape[0] for t in target_ids))
		
		input_ids_padded = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=self.tokenizer.pad_token_id) for t in input_ids])
		
		target_ids_padded = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=self.tokenizer.pad_token_id) for t in target_ids])
		
		attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()

		return input_ids_padded, attention_mask, target_ids_padded

	
	def _process_sequence(self, sequence):
		return f"".join(f"[{nucl.upper()}]" for nucl in sequence)

	def _process_target(self, builded_sequence):
		builded_sequence = re.sub(r"\(intron\)", "[INTRON]", builded_sequence, flags=re.IGNORECASE)
		builded_sequence = re.sub(r"\(exon\)", "[EXON]", builded_sequence, flags=re.IGNORECASE)

		def replace_nucleotides(match):
			segment = match.group(0)
			mapping = str.maketrans({"A": "[A]", "C": "[C]", "G": "[G]", "T": "[T]"})
			return segment.translate(mapping)

		pattern = r"(\[INTRON\]|\[EXON\])|([ACGT]+)"
		builded_sequence = re.sub(pattern, lambda m: m.group(1) if m.group(1) else replace_nucleotides(m), builded_sequence)

		return builded_sequence

	def _process_data(self, data):
		data["sequence"] = [self._process_sequence(sequence) for sequence in data["sequence"]]
		data["builded"] = [self._process_target(builded_sequence) for builded_sequence in data["builded"]]

		return data

	def add_train_data(self, data, batch_size=32, sequence_len=512):
		if sequence_len > 512:
			raise ValueError("cannot support sequences_len higher than 512")

		self._data_config = {
			"sequence_len": sequence_len,
			"batch_size": batch_size,
		}
		
		data = self._process_data(data)

		dataset = self.__RebuildSeqsGPTDataset__(data, self.tokenizer)

		self.train_dataset = dataset
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def _check_test_compatibility(self, sequence_len, batch_size):
		if hasattr(self, "_train_config"):
			if self._train_config["sequence_len"] != sequence_len or \
			self._train_config["batch_size"] != batch_size:
				print("Detected a different test dataloader configuration of the one used during training. This may lead to suboptimal results.")

	def add_test_data(self, data, batch_size=32, sequence_len=512):
		self._check_test_compatibility(sequence_len, batch_size)
		data = self._process_data(data)

		self.test_dataset = self.__RebuildSeqsGPTDataset__(data, self.tokenizer)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def train(self, lr=5e-5, epochs=3, save_at_end=True, save_freq=5):
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

				input_ids, attention_mask, target_ids = [b.to(self._device) for b in batch]
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
				
				loss = outputs.loss
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()

				if self.log_level == "info":
					train_bar.update(1)
					train_bar.set_postfix(loss=train_loss/train_bar.n)
	
			train_loss /= len(self.train_dataloader)
			if self.log_level == "info":
				train_bar.set_postfix({"Loss": train_loss})
				train_bar.close()
			history["train_loss"].append(train_loss)
		
			history["epoch"].append(epoch)

			if save_freq and (epoch+1) % save_freq == 0:
				self._save_checkpoint(epoch=epoch)
			
			self.epoch_end_time = time.time()
			history["time"].append(self.epoch_end_time - self.start_time)

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

		self.model.eval()
		with torch.no_grad():
			if self.log_level == "info":
				eval_bar = tqdm(self.test_dataloader, desc="Evaluating", leave=True)
			for batch in self.test_dataloader:
				input_ids, attention_mask, target_ids = [b.to(self._device) for b in batch]

				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
				loss = outputs.loss
				total_loss += loss.item()

				filtered_input_ids = [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask)]
				preds = []
				for filtered_input in filtered_input_ids:
					prediction = self.model.generate(
						input_ids=filtered_input.unsqueeze(0),
						attention_mask=torch.tensor([1]*filtered_input.size(-1)).unsqueeze(0).to(self._device),
						repetition_penalty=2.0,
						max_new_tokens=1024,
						pad_token_id=self.tokenizer.eos_token_id
					)

					preds.append(prediction[0][filtered_input.size(-1)])

				label_texts = [label[label != -100] for label in target_ids]

				for pred, label in zip(preds, label_texts):
					if pred == label:
						total_correct += 1

					total_samples += 1

				if self.log_level == "info":
					eval_bar.update(1)
					eval_bar.set_postfix(loss=total_loss/eval_bar.n)

		if self.log_level == "info":
			eval_bar.close()		
		avg_loss = total_loss / len(self.test_dataloader)
		overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

		print(f"Evaluation complete")
		print(f"Average loss: {avg_loss:.4f}")
		print(f"Overall Accuracy: {overall_accuracy:.4f}")

		self._eval_results = {
			"avg loss": avg_loss,
			"overall accuracy": overall_accuracy,
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
