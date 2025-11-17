

from typing import Any, Optional

import torch
from transformers import BertForSequenceClassification, Pipeline
from transformers.utils.generic import ModelOutput


def process_label(p: str) -> str:
	return "EXON" if p == 0 else "INTRON"

class ExInDNABERTPipeline(Pipeline):
	def _sanitize_parameters(
		self,
		**kwargs
	):
		preprocess_kwargs = {}

		for k in ("max_length"):
			if k in kwargs:
				preprocess_kwargs[k] = kwargs[k]
		
		forward_kwargs = {
			k: v for k, v in kwargs.items()
			if k not in preprocess_kwargs
		}

		postprocess_kwargs = {}

		return preprocess_kwargs, forward_kwargs, postprocess_kwargs

	def preprocess(
		self,
		input_,
		**preprocess_parameters
	):
		assert self.tokenizer

		if isinstance(input_, str):
			sequence = input_
		elif isinstance(input_, dict):
			sequence = input_.get("sequence", "")
		else:
				raise TypeError("input_ must be str or dict with 'sequence' key")

		sequence = sequence[:256]

		max_length = preprocess_parameters.get("max_length", 256)
		if not isinstance(max_length, int):
			raise TypeError("max_length must be an int")

		token_kwargs: dict[str, Any] = {"return_tensors": "pt"}
		token_kwargs["max_length"] = max_length
		token_kwargs["truncation"] = True

		enc = self.tokenizer(sequence, **token_kwargs).to(self.model.device)

		return {"prompt": sequence, "inputs": enc}
	
	def _forward(self, input_tensors: dict, **forward_params):
		assert isinstance(self.model, BertForSequenceClassification)
		kwargs = dict(forward_params)

		inputs = input_tensors.get("inputs")

		if inputs is None:
			raise ValueError("Model inputs missing in input_tensors (expected key 'inputs').")

		if hasattr(inputs, "items") and not isinstance(inputs, torch.Tensor):
			try:
				expanded_inputs: dict[str, torch.Tensor] = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}
			except Exception:
				expanded_inputs = {}
				for k, v in dict(inputs).items():
					expanded_inputs[k] = v.to(self.model.device) if isinstance(v, torch.Tensor) else v
		else:
			if isinstance(inputs, torch.Tensor):
				expanded_inputs = {"input_ids": inputs.to(self.model.device)}
			else:
				expanded_inputs = {"input_ids": torch.tensor(inputs, device=self.model.device)}

		self.model.eval()
		with torch.no_grad():
			outputs = self.model(**expanded_inputs, **kwargs)
		
		pred_id = torch.argmax(outputs.logits, dim=-1).item()

		return ModelOutput({"pred_id": pred_id})

	def postprocess(self, model_outputs: dict, **kwargs):
		assert self.tokenizer

		pred_id = model_outputs["pred_id"]
		return process_label(pred_id)