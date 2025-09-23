import re
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset
from llms.base import BaseModel
from schemas.train_params import TrainParams
from tqdm import tqdm
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from utils.data_collators import DataCollatorForFT
from utils.exceptions import MissingEssentialProp

valid_dna = set("ACGTURYSWKMBDHVN")
valid_prot = set("ACDEFGHIKLMNPQRSTVWY*X")

class Input(TypedDict):
	sequence: str
	target: str | None
	organism: str | None

class GenerateInput(TypedDict):
	sequence: str
	organism: str | None

class DnaTranslatorGPT(BaseModel):
	model: GPT2LMHeadModel | None = None
	tokenizer: GPT2Tokenizer | None = None
	max_length = 1024

	def load_checkpoint(
		self,
		checkpoint: str
	) -> None:
		model = GPT2LMHeadModel.from_pretrained(checkpoint)

		tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
		tokenizer.pad_token = tokenizer.eos_token

		special_tokens = ["[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]", "[DNA_R]", "[DNA_Y]", "[DNA_S]", "[DNA_W]", "[DNA_K]", "[DNA_M]", "[DNA_B]", "[DNA_D]", "[DNA_H]", "[DNA_V]", "[DNA_N]", "[PROT_A]", "[PROT_C]", "[PROT_D]", "[PROT_E]", "[PROT_F]", "[PROT_G]", "[PROT_H]", "[PROT_I]", "[PROT_K]", "[PROT_L]", "[PROT_M]", "[PROT_N]", "[PROT_P]", "[PROT_Q]", "[PROT_R]", "[PROT_S]", "[PROT_T]", "[PROT_V]", "[PROT_W]", "[PROT_Y]", "[PROT_*]", "[PROT_X]"]
		tokenizer.add_tokens(special_tokens)
		
		tokenizer.pad_token = "[PROT_*]"
		tokenizer.eos_token = "[PROT_*]"

		tokenizer.add_special_tokens({
			"eos_token": "[PROT_*]",
			"additional_special_tokens": ["<|DNA|>", "<|PROTEIN|>"]
		})

		tokenizer.add_eos_token = True

		model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

		self.model = model
		self.tokenizer = tokenizer

		if self.model is None or self.tokenizer is None:
			self._log("Error trying to load the checkpoint.", "WARNING")
			return None
		
		self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

	def from_pretrained(
		self,
		checkpoint: str
	) -> None:
		self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
		self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

	def _process_sequence(
		self,
		sequence: str
	) -> str:
		return "".join(f"[DNA_{nucl.upper()}]" for nucl in sequence if nucl.upper() in valid_dna)

	def _process_target(
		self,
		target: str
	) -> str:
		target = target + "*"
		target = target[:target.find("*") + 1]
		return "".join(f"[PROT_{prot.upper()}]" for prot in target if prot.upper() in valid_prot)
	
	def _unprocess_target(
		self,
		protein_tokens: str
	) -> str:
		matches = re.findall(r"\[PROT_([A-Z*])\]", protein_tokens.upper())
		return "".join(matches)
	
	def build_input(
		self,
		sequence: str,
		target: str | None = None,
		organism: str | None = None
	) -> Input:
		return {
			"sequence": sequence,
			"target": target,
			"organism": organism
		}
	
	def _build_input(
		self,
		sequence: str,
		target: str | None = None,
		organism: str | None = None
	) -> dict[Literal["partial", "complete"], str]:
			output = f"<|DNA|>{self._process_sequence(sequence)}"

			if organism:
				output += f"<|ORGANISM|>{organism[:10]}"

			output += f"<|PROTEIN|>"

			return {
				"partial": output,
				"complete": output+(self._process_target(target) if target else "")
			}	

	def _tokenize(
		self,
		input_text: str
	) -> tuple[torch.Tensor, torch.Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")

		tokenized = self.tokenizer(
			input_text,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt"
		).to(self.model.device)

		input_ids = tokenized["input_ids"]
		input_ids = cast(torch.Tensor, input_ids)
		attention_mask = tokenized["attention_mask"]
		attention_mask = cast(torch.Tensor, attention_mask)

		return (input_ids, attention_mask)

	def _tokenize_for_training(
		self,
		input_text: str,
		expected_text: str
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		encoded_input = self.tokenizer(
			input_text
		)
		encoded_expected = self.tokenizer(
			expected_text
		)

		input_ids = torch.tensor(encoded_expected["input_ids"], dtype=torch.long)
		attention_mask = torch.tensor(encoded_expected["attention_mask"], dtype=torch.bool)

		labels = torch.full_like(input_ids, -100)

		start = len(cast(list, encoded_input["input_ids"]))

		for i in range(start, len(input_ids)):
			if input_ids[i] != self.tokenizer.pad_token_id:
				labels[i] = input_ids[i]
		
		return input_ids, attention_mask, labels

	def _prepare_dataset(
		self,
		dataset: list[Input]
	) -> Dataset:
		tokenized = []

		for data in tqdm(dataset):
			promptfied = self._build_input(
				sequence=data["sequence"],
				target=data.get("target"),
				organism=data.get("organism", "")
			)
			tokenized_input = self._tokenize_for_training(
				input_text=promptfied["partial"],
				expected_text=promptfied["complete"]
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

		self._log("Training complete. You may save the model for later use.")

	def generate(
		self,
		input: GenerateInput
	) -> str:
		if self.model is None or self.tokenizer is None:
			raise MissingEssentialProp("Model or Tokenizer missing.")
		
		model_input = self._build_input(
			sequence=input["sequence"],
			organism=input.get("organism")
		)

		input_ids, attention_mask = self._tokenize(model_input["partial"])

		self.model.eval()
		with torch.no_grad():
			generated = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_new_tokens=128,
				pad_token_id=self.tokenizer.pad_token_id,
				do_sample=True,
				temperature=0.8,
				top_p=0.95,
				typical_p=0.98,
				num_beams=1
			)
		
		generated_texts = self.tokenizer.decode(generated[0], skip_special_tokens=False)

		start = generated_texts.find("<|PROTEIN|>")
		protein_tokenized = generated_texts[start + len("<|PROTEINS|>"):].strip()
		
		return self._unprocess_target(protein_tokenized)