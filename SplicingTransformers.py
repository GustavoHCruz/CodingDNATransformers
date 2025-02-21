import json
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch


class SplicingTransformers(ABC):
	def _set_seed(self, seed):
		self.seed = seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	
	def __init__(self, checkpoint, device="cuda", seed=None, notification=False, logs_dir="logs", models_dir="models", alias=None):
		self._device = device
		self.logs_dir = logs_dir
		self.models_dir = models_dir
		self.checkpoint = checkpoint
		self.alias = alias or checkpoint
		self.notification = notification

		self.model.to(self._device)

		print(f"Started {checkpoint} model")
	
	def _get_next_model_dir(self):
		os.makedirs(self.logs_dir, exist_ok=True)
		os.makedirs(self.models_dir, exist_ok=True)

		logs_dir = os.path.join(self.logs_dir, self.alias)
		model_dir = os.path.join(self.models_dir, self.alias)

		counter = 0
		if os.path.exists(logs_dir) or os.path.exists(model_dir):
			counter += 1

		logs_dir = f"{logs_dir}"
		model_dir = f"{model_dir}"
		
		if counter > 0:
			while os.path.exists(f"{logs_dir}_{counter}") or os.path.exists(f"{self.models_dir}_{counter}"):
				counter += 1
			
			logs_dir += f"_{counter}"
			model_dir += f"_{counter}"

		os.makedirs(f"{logs_dir}/checkpoints")

		self._logs_dir = logs_dir
		self._model_dir = model_dir

	def save_checkpoint(self, path=None):
		saving_path = self._model_dir
		if path:
			saving_path = path

		self.model.save_pretrained(saving_path)
		self.tokenizer.save_pretrained(saving_path)

		print(f"Model Successful Saved at {saving_path}")
	
	@abstractmethod
	def load_checkpoint(self, path):
		pass

	@abstractmethod
	def _collate_fn(self, batch):
		pass

	@abstractmethod
	def _process_sequence(self, sequence):
		pass

	@abstractmethod
	def _process_label(self, label):
		pass

	@abstractmethod
	def _process_data(self, data):
		pass

	@abstractmethod
	def add_train_data(self, data, batch_size, sequence_len, train_percentage, data_config):
		pass

	@abstractmethod
	def _check_test_compatibility(self, *args, **kargs):
		pass

	@abstractmethod
	def add_test_data(self, data, batch_size, sequence_len, data_config):
		pass
	
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
			path = f"{self._logs_dir}/checkpoints/checkpoint-epoch-{epoch}-"
		else:
			path = f"{self._logs_dir}/best-"

		torch.save(self.model.state_dict(), f"{path}-model.pth")
		torch.save(self.optimizer.state_dict(), f"{path}-optimizer.pth")

	def _load_checkpoint(self, epoch=None):
		if epoch != None:
			path = f"{self._logs_dir}/checkpoints/checkpoint-epoch-{epoch}-"
		else:
			path = f"{self._logs_dir}/best-"

		self.model.load_state_dict(torch.load(f"{path}-model.pth", map_location=self._device))
		self.optimizer.load_state_dict(torch.load(f"{path}-optimizer.pth", map_location=self._device))

	def _save_history(self, history):
		df = pd.DataFrame(history)
		df.to_csv(f"{self._logs_dir}/history.csv", index=False)
	
	def _save_config(self):
		with open(f"{self._logs_dir}/config.json", "w") as f:
			json.dump(self._train_config, f, indent=2)
	
	@abstractmethod
	def train(self, lr, epochs, evaluation, save_at_end, keep_best, save_freq):
		pass

	def _save_evaluation_results(self):
		with open(f"{self._logs_dir}/eval_results.json", "w") as f:
			json.dump(self._eval_results, f, indent=2)

	@abstractmethod
	def evaluate(self):
		pass

	@abstractmethod
	def _prediction_mapping(self, prediction):
		pass

	@abstractmethod
	def predict_single(self, data, map_pred):
		pass

	def predict_batch(self, data, map_pred=True):
		with torch.no_grad():
			return [self.predict_single(data=i, map_pred=map_pred) for i in data]
