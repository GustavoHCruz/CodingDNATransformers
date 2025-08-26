from typing import Any, Optional

from transformers.trainer_callback import (TrainerCallback, TrainerControl,
                                           TrainerState)
from transformers.training_args import TrainingArguments


class LogCollectorCallback(TrainerCallback):
	def __init__(self):
		self.logs: list[dict] = []
	
	def on_log(
		self,
		args: TrainingArguments,
		state: TrainerState,
		control: TrainerControl,
		logs: Optional[dict[str, Any]] = None,
		**kwargs
	):
		if logs is not None:
			self.logs.append(logs.copy())
