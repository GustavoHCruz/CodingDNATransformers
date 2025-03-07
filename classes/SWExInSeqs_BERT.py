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

class SWExInSeqsBERT(SplicingTransformers):
	class __SWExInBERT__(Dataset):
		def __init__(self, data, tokenizer, sequence_len, flanks_len):
			self.data = data
			self.tokenizer = tokenizer
			self.max_length = sequence_len + flanks_len * 2 + 75

		def __len__(self):
			return len(self.data["sequence"])
		
		def __getitem__(self, idx):
			prompt = f"Sequence:{self.data["sequence"][idx]}[SEP]"
			prompt = f"Flank Before: {self.data["flank_before"]}[SEP]"
			prompt = f"Flank After: {self.data["flank_after"]}[SEP]"
			prompt += f"Organism:{self.data["organism"][idx][:20]}[SEP]"
			
			prompt += "Answer:"

			input_ids = self.tokenizer.encode(prompt, max_length=self.max_length, padding=True, truncation=True)
			label = self.data["label"][idx]

			return torch.tensor(input_ids), torch.tensor(label)
	
	def __init__(self, checkpoint="bert-base-uncased", device="cuda", seed=None, notification=False,  logs_dir="logs", models_dir="models", alias=None, log_level="info"):
		if seed:
			self._set_seed(seed)

		self.log_level = log_level

		if checkpoint != "bert-base-uncased":
			self.load_checkpoint(checkpoint)
		else:
			self.model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
			self.tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)
		
		if checkpoint == "bert-base-uncased":
			special_tokens = ["[A]", "[C]", "[G]", "[T]", "[R]", "[Y]", "[S]", "[W]", "[K]", "[M]", "[B]", "[D]", "[H]", "[V]", "[N]", "[U]"]
			self.tokenizer.add_tokens(special_tokens)
			self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

		self.intron_token = 0
		self.exon_token = 1
		super().__init__(checkpoint=checkpoint, device=device, seed=seed, notification=notification, logs_dir=logs_dir, models_dir=models_dir, alias=alias)
		
	def load_checkpoint(self, path):
		self.model = BertForSequenceClassification.from_pretrained(path)
		self.tokenizer = BertTokenizer.from_pretrained(path)

	def _collate_fn(self, batch):
		return
	
	def _process_sequence(self, sequence):
		return
	
	def _process_target(self, label):
		return
	
	def _process_data(self, data):
		

		return data
	
	def add_train_data(self, data, batch_size=32, sequence_len=512, data_config=None):
		flanks_len = 10
		feat_hide_prob = 0.01
		if "flanks_len" in data_config:
			flanks_len = data_config["flanks_len"]
		if "feat_hide_prob" in data_config:
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

		dataset = self.__SpliceBERTDataset__(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)

		self.train_dataset = dataset
		
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

	def _check_test_compatibility(self, sequence_len, flanks_len, batch_size):
		if hasattr(self, "_train_config"):
			if self._train_config["sequence_len"] != sequence_len or \
			self._train_config["flanks_len"] != flanks_len or \
			self._train_config["batch_size"] != batch_size:
				print("Detected a different test dataloader configuration of the one used during training. This may lead to suboptimal results.")

	def add_test_data(self, data, batch_size=32, sequence_len=512, data_config=None):
		flanks_len = 10
		feat_hide_prob = 0.01
		if "flanks_len" in data_config:
			flanks_len = data_config["flanks_len"]
		if "feat_hide_prob" in data_config:
			feat_hide_prob = data_config["feat_hide_prob"]

		self._check_test_compatibility(sequence_len=sequence_len, flanks_len=flanks_len, batch_size=batch_size, feat_hide_prob=feat_hide_prob)

		data = self._process_data(data)

		self.test_dataset = self.__SpliceBERTDataset__(data, self.tokenizer, sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=feat_hide_prob)
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
			history["train_loss"].append(train_loss)
			if self.log_level == "info":
				train_bar.set_postfix({"Loss": train_loss})
				train_bar.close()

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
		

		self._eval_results = {}

		self._save_evaluation_results()

		if self.notification:
			notification.notify(title="Evaluation complete", timeout=5)
	
	def _prediction_mapping(self, prediction):
		return 
	
	def predict_single(self, data, map_pred=True):
		if map_pred:
			return self._prediction_mapping()
		
		return 