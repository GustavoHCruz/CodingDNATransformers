import re
from typing import TypedDict, cast

import torch
from datasets import Dataset
from llms.base import BaseModel
from schemas.train_params import TrainParams
from torch import Tensor
from tqdm import tqdm
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from utils.data_collators import DataCollatorForFT
from utils.exceptions import MissingEssentialProp


class Input(TypedDict):
	dna_sequence: str
	protein_sequence: str | None
	organism: str | None

class DNATranslatorGPT(BaseModel):
	model: GPT2LMHeadModel | None = None
	tokenizer: GPT2Tokenizer | None = None
	max_length = 1024

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		model = GPT2LMHeadModel.from_pretrained(
			checkpoint,
			device_map="auto"
		)
		tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

		special_tokens = [
			"[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]", "[DNA_R]",
			"[DNA_Y]", "[DNA_S]", "[DNA_W]", "[DNA_K]", "[DNA_M]",
			"[DNA_B]", "[DNA_D]", "[DNA_H]", "[DNA_V]", "[DNA_N]", 
			"[PROT_A]", "[PROT_C]", "[PROT_D]", "[PROT_E]",
			"[PROT_F]", "[PROT_G]", "[PROT_H]", "[PROT_I]",
			"[PROT_K]", "[PROT_L]", "[PROT_M]", "[PROT_N]",
			"[PROT_P]", "[PROT_Q]", "[PROT_R]", "[PROT_S]",
			"[PROT_T]", "[PROT_V]", "[PROT_W]", "[PROT_Y]",
			"[PROT_*]", "[PROT_X]"
		]
		tokenizer.add_tokens(special_tokens)
		tokenizer.add_special_tokens({
			"additional_special_tokens": ["<|DNA|>", "<|ORGANISM|>", "<|PROTEIN|>", "<|END|>"]
		})

		tokenizer.pad_token = tokenizer.eos_token
		model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

		if model is None or tokenizer is None:
			self._log("Error trying to load the checkpoint.", "WARNING")
			return None
		
		self.model = model
		self.tokenizer = tokenizer
		
	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = GPT2LMHeadModel.from_pretrained(
			checkpoint,
			device_map="auto"
		)
		self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

	def _preprare_dna_sequence(
		self,
		sequence: str
	) -> str:
		return "".join(f"[DNA_{nucl.upper()}]" for nucl in sequence)

	def _prepare_protein_sequence(
		self,
		sequence: str
	) -> str:
		return "".join(f"[PROT_{amino_acid.upper()}]" for amino_acid in sequence)
	
	def build_input(
		self,
		dna_sequence: str,
		protein_sequence: str | None = None,
		organism: str | None = None
	) -> Input:
		return {
			"dna_sequence": dna_sequence,
			"protein_sequence": protein_sequence,
			"organism": organism
		}
	
	def _build_input(
		self,
		data: Input
	) -> tuple[str, str | None]:
		input_sequence = f"<|DNA|>{self._preprare_dna_sequence(data['dna_sequence'])}"

		organism = data.get("organism")
		if organism:
			input_sequence += f"<|ORGANISM|>{organism[:10].lower().strip()}"

		output_sequence = None

		protein = data.get("protein_sequence")
		if protein:
			output_sequence = f"<|PROTEIN|>{self._prepare_protein_sequence(protein)}<|END|>"
		
		return input_sequence, output_sequence

	def _tokenize_for_inference(
		self,
		input_sequence: str
	) -> tuple[Tensor, Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")

		tokenized = self.tokenizer(
			input_sequence,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		).to(self.model.device)

		input_ids = tokenized["input_ids"]
		assert isinstance(input_ids, Tensor)
		attention_mask = tokenized["attention_mask"]
		assert isinstance(attention_mask, Tensor)

		return (input_ids, attention_mask)
	
	def _tokenize_for_training(
		self,
		input_text: str,
		expected_text: str
	) -> tuple[Tensor, Tensor, Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		encoded_input = self.tokenizer(input_text)
		only_inputs = encoded_input["input_ids"]
		assert isinstance(only_inputs, list)

		encoded = self.tokenizer(input_text + expected_text)

		input_ids = torch.tensor(encoded["input_ids"])
		attention_mask = torch.tensor(encoded["attention_mask"])

		labels = input_ids.clone()

		labels[:len(only_inputs)] = -100
		
		return input_ids, attention_mask, labels

	def _prepare_dataset(
		self,
		dataset: list[Input]
	) -> Dataset:
		tokenized = []

		for data in tqdm(dataset):
			input_sequence, output_sequence = self._build_input(data)

			if output_sequence is None:
				raise ValueError("Target is missing.")

			tokenized_input = self._tokenize_for_training(
				input_text=input_sequence,
				expected_text=output_sequence
			)

			input_ids, attention_mask, labels = tokenized_input
			tokenized.append({
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"labels": labels
			})

		return Dataset.from_list(tokenized)

	def train(
		self,
		dataset: list[Input],
		params: TrainParams
	) -> None:
		if not self.model or not self.tokenizer:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		self._log("Preparing dataset...")
		data = self._prepare_dataset(dataset)
		self._log("Dataset prepared!")

		args = TrainingArguments(
			num_train_epochs=params.epochs,
			optim=params.optim,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			lr_scheduler_type="cosine",
			save_strategy="no",
			logging_steps=params.logging_steps,
		)

		if self.seed:
			args.seed = self.seed

		trainer = Trainer(
			model=self.model,
			train_dataset=data,
			args=args,
			data_collator=DataCollatorForFT(self.tokenizer),
		)

		self._log("Starting training...")

		trainer.train()

		self._log("Training complete. You may save the model for later usage.")

	def _unprocess_target(
		self,
		protein_tokens: str
	) -> str:
		matches = re.findall(r"\[PROT_([A-Z*])\]", protein_tokens.upper())
		return "".join(matches)

	def generate(
		self,
		input: Input
	) -> str:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		input_sequence, _ = self._build_input(input)
		
		input_ids, attention_mask = self._tokenize_for_inference(input_sequence)
		
		max_new_tokens = self.max_length - len(input_ids[0])

		self.model.eval()
		with torch.no_grad():
			generated = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=max_new_tokens,
				pad_token_id=self.tokenizer.pad_token_id,
				eos_token_id=self.tokenizer.convert_tokens_to_ids("<|END|>"),
				do_sample=True,
				temperature=0.8,
				top_p=0.95,
				typical_p=0.98,
				num_beams=1,
				repetition_penalty=1.1
			)
		
		generated_texts = self.tokenizer.decode(generated[0], skip_special_tokens=False)

		start = generated_texts.find("<|PROTEIN|>") + len("<|PROTEIN|>")
		end = generated_texts.find("<|END|>", start)
		protein_tokenized = generated_texts[start:end].strip()

		return self._unprocess_target(protein_tokenized)