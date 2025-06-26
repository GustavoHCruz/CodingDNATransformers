import csv
import logging
import time
from math import ceil
from typing import Any, Generator

import torch
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

valid_dna = set("ACGTURYSWKMBDHVN")
valid_prot = set("ACDEFGHIKLMNPQRSTVWY*X")

def process_sequence(sequence: str) -> str:
  return "".join(f"[DNA_{nucl.upper()}]" for nucl in sequence if nucl.upper() in valid_dna)

def process_target(target: str) -> str:
  target = target + "*"
  target = target[:target.find("*") + 1]
  return "".join(f"[PROT_{prot.upper()}]" for prot in target if prot.upper() in valid_prot)

def promptfy(dna_tokens: str, protein_tokens=None) -> str:
  if protein_tokens:
    return f"<|DNA|> {dna_tokens} <|PROTEIN|> {protein_tokens}"
  return f"<|DNA|> {dna_tokens} <|PROTEIN|>"

class DNADatasetFinetune(IterableDataset):
  def __init__(self, csv_path: str, tokenizer, dataset_total_length: int, sequence_max_length=1024) -> None:
    self.csv_path = csv_path
    self.tokenizer = tokenizer
    self.max_length = sequence_max_length
    self._length = dataset_total_length
  
  def __len__(self) -> int:
    return self._length

  def __iter__(self) -> Generator[dict[str, torch.Tensor], Any, None]:
    with open(self.csv_path, newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        seq = process_sequence(row["sequence"])
        tgt = process_target(row["target"])

        partial = promptfy(seq)
        full = promptfy(seq, tgt)

        partial_encoded = self.tokenizer(partial)
        full_encoded = self.tokenizer(
          full,
          truncation=True,
          padding="max_length",
          max_length=self.max_length
        )

        input_ids = full_encoded["input_ids"]
        attention_mask = full_encoded["attention_mask"]

        labels = [-100] * len(input_ids)
        start = min(len(partial_encoded["input_ids"]), len(input_ids))

        for i in range(start, len(input_ids)):
          if input_ids[i] != self.tokenizer.pad_token_id:
            labels[i] = input_ids[i]

        yield {
          "input_ids": torch.tensor(input_ids),
          "attention_mask": torch.tensor(attention_mask),
          "labels": torch.tensor(labels)
        }

class FinetuneDataCollator:
  def __init__(self, tokenizer) -> None:
    self.tokenizer = tokenizer
    self.pad_token_id = tokenizer.pad_token_id

  def __call__(self, batch) -> dict[str, torch.Tensor]:
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["labels"] for example in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
      "input_ids": input_ids_padded,
      "attention_mask": attention_mask_padded,
      "labels": labels_padded
    }

def load_checkpoint(path: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
  model = AutoModelForCausalLM.from_pretrained(path)
  tokenizer = AutoTokenizer.from_pretrained(path)
  special_tokens = ["[DNA_A]", "[DNA_C]", "[DNA_G]", "[DNA_T]", "[DNA_R]", "[DNA_Y]", "[DNA_S]", "[DNA_W]", "[DNA_K]", "[DNA_M]", "[DNA_B]", "[DNA_D]", "[DNA_H]", "[DNA_V]", "[DNA_N]", "[PROT_A]", "[PROT_C]", "[PROT_D]", "[PROT_E]", "[PROT_F]", "[PROT_G]", "[PROT_H]", "[PROT_I]", "[PROT_K]", "[PROT_L]", "[PROT_M]", "[PROT_N]", "[PROT_P]", "[PROT_Q]", "[PROT_R]", "[PROT_S]", "[PROT_T]", "[PROT_V]", "[PROT_W]", "[PROT_Y]", "[PROT_*]", "[PROT_X]"]
  tokenizer.add_tokens(special_tokens)

  tokenizer.pad_token = "[PROT_*]"
  tokenizer.eos_token = "[PROT_*]"

  tokenizer.add_special_tokens({
      "eos_token": "[PROT_*]",
      "additional_special_tokens": ["<|DNA|>", "<|PROTEIN|>"]
  })

  tokenizer.padding_side = "left"
  tokenizer.add_eos_token = True

  model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

  return model, tokenizer

def train(accelerator: Accelerator, model: PreTrainedModel, train_dataloader: DataLoader) -> tuple[PreTrainedModel, dict[str, Any]]:
  epochs = 1
  lr = 1e-5
  gradient_accumulation_steps = 8
  optimizer = AdamW(model.parameters(), lr=lr)

  num_training_steps = epochs * len(train_dataloader)
  num_warmup_steps = int(0.03 * num_training_steps)

  lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
  )

  model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    lr_scheduler
  )

  train_dataloder_len = len(train_dataloader)
  history = {"epoch": [], "time": [], "train_loss": [], "lr": []}
  start_time = time.time()

  train_bar = tqdm(total=ceil(num_training_steps/gradient_accumulation_steps), desc=f"Steps", leave=True, disable=not accelerator.is_local_main_process)
  global_step = 0

  model.train()
  for epoch in range(epochs):
    train_loss = 0.0

    accumulated_loss = 0.0
    for batch_idx, batch in enumerate(train_dataloader):
      outputs = model(**batch)
      loss = outputs.loss / gradient_accumulation_steps

      accelerator.backward(loss)
      accumulated_loss += loss.item()

      if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1 == train_dataloder_len):
        accelerator.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        train_loss += loss_val
        global_step += 1

        current_epoch_fraction = epoch + (batch_idx / train_dataloder_len)
        current_lr = lr_scheduler.get_last_lr()[0]

        train_bar.update(gradient_accumulation_steps)
        train_bar.set_postfix(loss=loss_val, lr=current_lr, epoch=round(current_epoch_fraction, 2))

        history["epoch"].append(round(current_epoch_fraction, 2))
        history["train_loss"].append(accumulated_loss)
        history["lr"].append(current_lr)
        history["time"].append(time.time() - start_time)

        accumulated_loss = 0.0

    train_bar.close()

  return model, history

def main() -> None:
  train_csv_path = "dna_proteins.csv"
  checkpoint = "gpt2"
  output_path = "./ProtGPT"

  accelerator = Accelerator()
  is_main_process = accelerator.is_main_process
  num_gpus = accelerator.num_processes
  logger = logging.getLogger(__name__)

  logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

  if is_main_process:
    logger.info(f"Selected Model: {checkpoint}")

  model, tokenizer = load_checkpoint(checkpoint)

  train_dataset = DNADatasetFinetune(csv_path=train_csv_path, tokenizer=tokenizer, dataset_total_length=681899)
  train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=FinetuneDataCollator(tokenizer))

  if is_main_process:
    logger.info(f"Starting fine-tune with {num_gpus} GPU(s)")

  model, _ = train(accelerator, model, train_dataloader)
  
  model = accelerator.unwrap_model(model)

  if is_main_process:
    logger.info(f"Saving finetuned model at {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
  
  if is_main_process:
    logger.info(f"Finishing execution")

  accelerator.wait_for_everyone()
  accelerator.end_training()

if __name__ == "__main__":
    main()