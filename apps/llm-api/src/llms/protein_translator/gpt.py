import pandas as pd
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

seed = 1234

checkpoint = "gpt2"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")

special_tokens = ["[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]", "[DNA_R]", "[DNA_Y]", "[DNA_S]", "[DNA_W]", "[DNA_K]", "[DNA_M]", "[DNA_B]", "[DNA_D]", "[DNA_H]", "[DNA_V]", "[DNA_N]", "[PROT_A]", "[PROT_C]", "[PROT_D]", "[PROT_E]", "[PROT_F]", "[PROT_G]", "[PROT_H]", "[PROT_I]", "[PROT_K]", "[PROT_L]", "[PROT_M]", "[PROT_N]", "[PROT_P]", "[PROT_Q]", "[PROT_R]", "[PROT_S]", "[PROT_T]", "[PROT_V]", "[PROT_W]", "[PROT_Y]", "[PROT_*]", "[PROT_X]"]
tokenizer.add_special_tokens({
    "eos_token": "[PROT_*]",
    "additional_special_tokens": ["<|DNA|>", "<|PROTEIN|>"]
})
tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

tokenizer.pad_token = "PROT_*"
tokenizer.eos_token = "PROT_*"
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

df = pd.read_csv("dna_proteins.csv")

df = df.drop(["id", "flankBefore", "flankAfter", "gene", "organism", "parentDatasetId"], axis=1)

def process_sequence(sequence: str) -> str:
  return f"".join(f"[DNA_{nucl.upper()}]" for nucl in sequence)

def process_target(target: str) -> str:
	target = target+"*"
	target = target[0:target.find("*")+1]
	return f"".join(f"[PROT_{prot.upper()}]" for prot in target)

df["sequence"] = df["sequence"].apply(process_sequence)
df["target"] = df["target"].apply(process_target)

data = df.to_dict(orient="records")

def promptfy(dna_tokens, protein_tokens=None) -> str:
	if protein_tokens:
		return f"<|DNA|> {dna_tokens} <|PROTEIN|> {protein_tokens}"
	return f"<|DNA|> {dna_tokens} <|PROTEIN|>"


tokenized_data = []
for d in data:
	partial = promptfy(d["sequence"])
	full = promptfy(d["sequence"], d["target"])

	partial_encoded = tokenizer(partial)
	full_encoded = tokenizer(full, padding="max_length", truncation=True, max_length=1024)

	input_ids = full_encoded["input_ids"]
	attention_mask = full_encoded["attention_mask"]

	labels = [-100] * len(input_ids)
	start = min(len(partial_encoded["input_ids"]), 1024)

	for i in range(start, len(input_ids)):
		if input_ids[i] != tokenizer.pad_token_type_id:
			labels[i] = input_ids[i]
	
	tokenized_data.append({
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"labels": labels
	})

del data
del df

dataset = Dataset.from_list(tokenized_data)

del tokenized_data

train_dataset, eval_dataset = dataset.train_test_split(test_size=0.1, seed=seed).values()

del dataset

class DataCollator:
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		self.pad_token_id = tokenizer.pad_token_id
	
	def __call__(self, batch):
		input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
		attention_mask = [torch.tensor(example["attention_mask"], dtype=torch.bool) for example in batch]
		labels = [torch.tensor(example["labels"], dtype=torch.long) for example in batch]

		input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
		attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
		labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

		return {
			"input_ids": input_ids_padded,
			"attention_mask": attention_mask_padded,
			"labels": labels_padded
		}
	
model.enable_input_require_grads()
model.config.use_cache = False

trainer = Trainer(
  model=model,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  args=TrainingArguments(
    per_device_train_batch_size=2,
		gradient_accumulation_steps=8,
		learning_rate=2e-5,
		num_train_epochs=1,
		lr_scheduler_type="cosine",
		logging_steps=1,
		bf16=True,
		optim="adamw_torch",
		label_names=["labels"],
		seed=seed,
		warmup_ratio=0.03,
		eval_strategy="epoch"
	),
  data_collator=DataCollator(tokenizer=tokenizer)
)

trainer.train()

model.save_pretrained("./")
tokenizer.save_pretrained("./")