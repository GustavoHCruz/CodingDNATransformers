import csv
import re
from typing import Any, Generator

import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, IterableDataset
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

valid_dna = set("ACGTURYSWKMBDHVN")
valid_prot = set("ACDEFGHIKLMNPQRSTVWY*X")

def process_sequence(sequence: str) -> str:
  return "".join(f"[DNA_{nucl.upper()}]" for nucl in sequence if nucl.upper() in valid_dna)

def process_target(target: str) -> str:
  target = target + "*"
  target = target[:target.find("*") + 1]
  return "".join(f"[PROT_{prot.upper()}]" for prot in target if prot.upper() in valid_prot)

def promptfy(dna_tokens: str) -> str:
  return f"<|DNA|> {dna_tokens} <|PROTEIN|>"

def unprocess_target(protein_tokens: str) -> str:
    matches = re.findall(r"\[PROT_([A-Z*])\]", protein_tokens.upper())
    return "".join(matches)

class DNADatasetEvaluation(IterableDataset):
  def __init__(self, csv_path: str, tokenizer, dataset_total_length: int, sequence_max_length=512) -> None:
    self.csv_path = csv_path
    self.tokenizer = tokenizer
    self.max_length = sequence_max_length
    self._length = dataset_total_length
  
  def __len__(self) -> int:
    return self._length

  def __iter__(self) -> Generator[dict[str, torch.Tensor | str], Any, None]:
    with open(self.csv_path, newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        seq = process_sequence(row["sequence"])
        tgt = process_target(row["target"])

        prompt = promptfy(seq)

        prompt_encoded = self.tokenizer(
          prompt,
          truncation=True,
          padding="max_length",
          max_length=self.max_length
        )
        
        tgt_encoded = self.tokenizer(
          tgt
        )

        input_ids = prompt_encoded["input_ids"]
        attention_mask = prompt_encoded["attention_mask"]
        labels = tgt_encoded["input_ids"]

        yield {
          "input_ids": torch.tensor(input_ids),
          "attention_mask": torch.tensor(attention_mask),
          "labels": torch.tensor(labels)
        }

def load_finetuned(path: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
  model = AutoModelForCausalLM.from_pretrained(path)
  tokenizer = AutoTokenizer.from_pretrained(path)

  return model, tokenizer

eval_csv_path = "eval.csv"
checkpoint = "ProtGPT"

model, tokenizer = load_finetuned(checkpoint)

eval_dataset = DNADatasetEvaluation(csv_path=eval_csv_path, tokenizer=tokenizer, dataset_total_length=100)
eval_dataloader = DataLoader(eval_dataset, batch_size=1)

model.eval()

preds = []
refs = []

for batch in eval_dataloader:
  with torch.no_grad():
    generated_ids = model.generate(
      input_ids=batch["input_ids"],
      attention_mask=batch["attention_mask"],
      max_new_tokens=128,
      pad_token_id=tokenizer.pad_token_id,
      do_sample=True,
      temperature=0.8,
      top_p=0.95,
      typical_p=0.98,
      num_beams=1
    )
    
  generated_texts = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

  start = generated_texts.find("<|PROTEIN|>")
  protein_tokenized = generated_texts[start + len("<|PROTEIN|>"):].strip()
  preds.append(unprocess_target(protein_tokenized))
  decoded_tgt = tokenizer.decode(batch["labels"][0])
  refs.append(unprocess_target(decoded_tgt))

  print("pred", unprocess_target(protein_tokenized))
  print("ref", unprocess_target(decoded_tgt))

rouge_metrics = {'rouge1': [], 'rouge2': [], 'rougeL': []}
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

for ref, pred in zip(refs, preds):
  scores = scorer.score(ref, pred)
  for key in rouge_metrics:
    rouge_metrics[key].append(scores[key].fmeasure)

rouge_avg = {k: sum(v)/len(v) for k, v in rouge_metrics.items()}

print(rouge_avg)