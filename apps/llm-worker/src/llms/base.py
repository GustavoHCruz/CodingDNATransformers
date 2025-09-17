import json
import logging
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch
from colorama import Fore, Style


class BaseModel(ABC):
	model = None
	tokenizer = None
	seed = None

	def __init__(
		self,
		checkpoint: str | None = None,
		log_level = "INFO",
		seed: int | None = None
	) -> None:
		if seed:
			self._set_seed(seed)
		
		self._set_logger(
			level=log_level
		)
		
		if checkpoint:
			self._checkpoint = checkpoint
			self.load_checkpoint(
				checkpoint=checkpoint
			)
	
	def _set_seed(
		self,
		seed: int
	) -> None:
		self.seed = seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True)

		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
		os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
	
	def _set_logger(
		self,
		level: str
	) -> None:
		logging.basicConfig(
			level=level,
			format="%(asctime)s - %(levelname)s - %(message)s"
		)
		self.logger = logging.getLogger(__name__)
	
	def _log(
		self,
		info: str,
		level: Literal["INFO", "WARNING", "ERROR", "DEBUG"] = "INFO"
	) -> None:
		if level == "INFO":
			colored_msg = f"{Fore.GREEN}{info}{Style.RESET_ALL}"
			self.logger.info(colored_msg)
		elif level == "WARNING":
			colored_msg = f"{Fore.YELLOW}{info}{Style.RESET_ALL}"
			self.logger.warning(colored_msg)
		elif level == "ERROR":
			colored_msg = f"{Fore.RED}{info}{Style.RESET_ALL}"
			self.logger.error(colored_msg)
		elif level == "DEBUG":
			colored_msg = f"{Fore.BLUE}{info}{Style.RESET_ALL}"
			self.logger.debug(colored_msg)
	 
	def set_seed(
		self,
		seed: int
	) -> None:
		self._log(f"Setting seed value to {seed}")
		self._set_seed(seed)
		self._log(f"Seed setted")
	
	@abstractmethod
	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		pass

	@abstractmethod
	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		pass

	def save_pretrained(
		self,
		output_path: str
	) -> None:
		self._log(f"Attempting to save model at '{output_path}'")
		if self.model is None or self.tokenizer is None:
			self._log("Model or Tokenizer not found. Aborting...")
			return None

		self.model.save_pretrained(output_path)
		self.tokenizer.save_pretrained(output_path)
		self._log(f"Successfully saved at '{output_path}'")

	def unload_model(self) -> None:
		try:
			self._log("Trying to reset model...")
			del self.model
			self.model = None
		except Exception as e:
			self._log("Couldn't reset model...", "ERROR")
			self._log(str(e), "DEBUG")
	
	def __del__(self) -> None:
		self.unload_model