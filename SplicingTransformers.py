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
	
	def __init__(self, checkpoint, device="cuda", seed=None, notification=False, logs_dir="logs", alias=None):
		self._device = device
		self.logs_dir = logs_dir
		self.checkpoint = checkpoint
		self.alias = alias or checkpoint
		self.notification = notification

		self.model.to(self._device)

		if seed:
			self._set_seed(seed)
	
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
	def add_train_data(self, data, batch_size, sequence_len, *args, **kargs):
		pass

	@abstractmethod
	def _check_test_compatibility(self, *args, **kargs):
		pass

	@abstractmethod
	def add_test_data(self, data, batch_size, *args, **kargs):
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
			path = f"{self._model_dir}/checkpoints/checkpoint-epoch-{epoch}-"
		else:
			path = f"{self._model_dir}/best-"

		torch.save(self.model.state_dict(), f"{path}-model.pth")
		torch.save(self.optimizer.state_dict(), f"{path}-optimizer.pth")

	def _load_checkpoint(self, epoch=None):
		if epoch != None:
			path = f"{self._model_dir}/checkpoints/checkpoint-epoch-{epoch}-"
		else:
			path = f"{self._model_dir}/best-"

		self.model.load_state_dict(torch.load(f"{path}-model.pth", map_location=self._device))
		self.optimizer.load_state_dict(torch.load(f"{path}-optimizer.pth", map_location=self._device))

	def _save_history(self, history):
		df = pd.DataFrame(history)
		df.to_csv(f"{self._model_dir}/history.csv", index=False)
	
	def _save_config(self):
		with open(f"{self._model_dir}/config.json", "w") as f:
			json.dump(self._train_config, f, indent=2)
	
	@abstractmethod
	def train(self, lr, epochs, evaluation, save_at_end, keep_best, save_freq):
		pass

	@abstractmethod
	def _prediction_mapping(self, prediction):
		pass

	@abstractmethod
	def predict_single(self, data, map_pred):
		pass

	def predict_batch(self, data, map_pred=True):
		preds = []
		for i in data:
			pred = self.predict_single(data=i, map_pred=map_pred)
			preds.append(pred)

		return preds