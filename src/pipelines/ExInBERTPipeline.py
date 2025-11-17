

from typing import Any, Optional

import torch
from transformers import BertForSequenceClassification, Pipeline
from transformers.utils.generic import ModelOutput

DNA_MAP = {
	"A": "[A]",
	"C": "[C]",
	"G": "[G]",
	"T": "[T]",
	"R": "[R]",
	"Y": "[Y]",
	"S": "[S]",
	"W": "[W]",
	"K": "[K]",
	"M": "[M]",
	"B": "[B]",
	"D": "[D]",
	"H": "[H]",
	"V": "[V]",
	"N": "[N]"
}

def process_sequence(seq: str) -> str:
	seq = seq.strip().upper()
	return "".join(DNA_MAP.get(ch, "[N]") for ch in seq)

def process_label(p: str) -> str:
	return "EXON" if p == 0 else "INTRON"

def ensure_optional_str(value: Any) -> Optional[str]:
	return value if isinstance(value, str) else None

class ExInBERTPipeline(Pipeline):
	def _build_prompt(
		self,
		sequence: str,
		organism: Optional[str],
		gene: Optional[str],
		before: Optional[str],
		after: Optional[str]
	) -> str: 
		out = f"<|SEQUENCE|>{process_sequence(sequence[:256])}"
		
		if organism:
			out += f"<|ORGANISM|>{organism[:10]}"
		
		if gene:
			out += f"<|GENE|>{gene[:10]}"

		if before:
			before_p = process_sequence(before[:25])
			out += f"<|FLANK_BEFORE|>{before_p}"
		
		if after:
			after_p = process_sequence(after[:25])
			out += f"|<FLANK_AFTER|>{after_p}"
		
		return out

	def _sanitize_parameters(
		self,
		**kwargs
	):
		preprocess_kwargs = {}

		for k in ("organism", "gene", "before", "after", "max_length"):
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

		organism_raw = preprocess_parameters.get("organism", None)
		gene_raw = preprocess_parameters.get("gene", None)
		before_raw = preprocess_parameters.get("before", None)
		after_raw = preprocess_parameters.get("after", None)

		if organism_raw is None and isinstance(input_, dict):
			organism_raw = input_.get("organism", None)
		if gene_raw is None and isinstance(input_, dict):
			gene_raw = input_.get("gene", None)
		if before_raw is None and isinstance(input_, dict):
			before_raw = input_.get("before", None)
		if after_raw is None and isinstance(input_, dict):
			after_raw = input_.get("after", None)

		organism: Optional[str] = ensure_optional_str(organism_raw)
		gene: Optional[str] = ensure_optional_str(gene_raw)
		before: Optional[str] = ensure_optional_str(before_raw)
		after: Optional[str] = ensure_optional_str(after_raw)

		max_length = preprocess_parameters.get("max_length", 256)
		if not isinstance(max_length, int):
			raise TypeError("max_length must be an int")

		prompt = self._build_prompt(sequence, organism, gene, before, after)

		token_kwargs: dict[str, Any] = {"return_tensors": "pt"}
		token_kwargs["max_length"] = max_length
		token_kwargs["truncation"] = True

		enc = self.tokenizer(prompt, **token_kwargs).to(self.model.device)

		return {"prompt": prompt, "inputs": enc}
	
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