import re
from typing import TypedDict

import torch
from datasets import Dataset
from llms.base import BaseModel
from schemas.train_params import TrainParams
from tqdm import tqdm
from transformers import (BatchEncoding, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5ForConditionalGeneration, T5Tokenizer)
from utils.exceptions import MissingEssentialProp


class Input(TypedDict):
	dna_sequence: str
	protein_sequence: str | None
	organism: str | None

class DNATranslatorT5(BaseModel):
	model: T5ForConditionalGeneration | None = None
	tokenizer: T5Tokenizer | None = None
	max_length = 512

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		model = T5ForConditionalGeneration.from_pretrained(checkpoint)
		tokenizer = T5Tokenizer.from_pretrained(checkpoint)

		special_tokens = [
			"[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]", "[DNA_R]",
			"[DNA_Y]", "[DNA_S]", "[DNA_W]", "[DNA_K]", "[DNA_M]",
			"[DNA_B]", "[DNA_D]", "[DNA_H]", "[DNA_V]", "[DNA_N]", 
			"[PROT_A]", "[PROT_C]", "[PROT_D]", "[PROT_E]",
			"[PROT_F]", "[PROT_G]", "[PROT_H]", "[PROT_I]",
			"[PROT_K]", "[PROT_L]", "[PROT_M]", "[PROT_N]",
			"[PROT_P]", "[PROT_Q]", "[PROT_R]", "[PROT_S]",
			"[PROT_T]", "[PROT_V]", "[PROT_W]", "[PROT_Y]",
			"[PROT_X]"
		]
		tokenizer.add_tokens(special_tokens)
		tokenizer.add_special_tokens({
			"additional_special_tokens": ["<|DNA|>", "<|ORGANISM|>", "<|PROTEIN|>"]
		})

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
		self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
		self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)

	def _preprare_dna_sequence(
		self,
		sequence: str
	) -> str:
		return f"".join(f"[DNA_{nucl.upper()}]" for nucl in sequence)
	
	def _prepare_protein_sequence(
		self,
		sequence: str
	) -> str:
		return f"".join(f"[PROT_{amino_acid}]" for amino_acid in sequence)
	
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
			output_sequence = f"<|PROTEIN|>{self._prepare_protein_sequence(protein)}"

		return input_sequence, output_sequence
	
	def _tokenize_for_inference(
		self,
		dna_sequence: str
	) -> BatchEncoding:
		if self.tokenizer is None or self.model is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")

		tokenized_input = self.tokenizer(
			dna_sequence,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		).to(self.model.device)

		return tokenized_input

	def _tokenize_for_training(
		self,
		input_sequence: str,
		output_sequence: str
	) -> BatchEncoding:
		if self.tokenizer is None:
			raise MissingEssentialProp("Tokenizer missing.")
		
		tokenized_input = self.tokenizer(
			input_sequence,
			text_target=output_sequence,
			truncation=True,
			max_length=self.max_length
		)

		return tokenized_input
		
	def _prepare_dataset(
		self,
		dataset: list[Input]
	) -> Dataset:
		tokenized_dataset = []

		for data in tqdm(dataset):
			input_sequence, output_sequence = self._build_input(data)

			if output_sequence is None:
				raise ValueError("Target is missing.")

			tokenized_input = self._tokenize_for_training(
				input_sequence=input_sequence,
				output_sequence=output_sequence
			)

			tokenized_dataset.append(tokenized_input)

		return Dataset.from_list(tokenized_dataset)

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

		args = Seq2SeqTrainingArguments(
			num_train_epochs=params.epochs,
			optim=params.optim,
			learning_rate=params.lr,
			per_device_train_batch_size=params.batch_size,
			gradient_accumulation_steps=params.gradient_accumulation,
			lr_scheduler_type="cosine",
			save_strategy="no",
			logging_steps=params.logging_steps,
			predict_with_generate=True
		)

		if self.seed:
			args.seed = self.seed
		
		collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

		trainer = Seq2SeqTrainer(
			model=self.model,
			train_dataset=data,
			args=args,
			data_collator=collator
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
		data: Input
	) -> str:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		sentence, _ = self._build_input(data)

		tokenized = self._tokenize_for_inference(sentence)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model.generate(
				input_ids=tokenized["input_ids"],
				num_beams=3,
				early_stopping=True
			)
		
		return self._unprocess_target(self.tokenizer.decode(outputs[0], skip_special_tokens=True))