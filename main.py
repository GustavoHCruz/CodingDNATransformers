import pandas as pd

from BERT_based import SpliceBERT, SpliceDNABERT
from GPT_based import SpliceGPT

config = {
  "name": "bert-001",
  "checkpoint_default": True,
  "checkpoint": "bert-base-uncased",
  "train_dataset": "100k",
  "train_percentage": 1.0,
  "test_dataset": "3k",
  "dataset_version": "small",
  "seed": 1234,
  "batch_size": 32,
  "hide_prob": 0.4,
  "lr": 5e-5,
  "epochs": 5
}

if config["checkpoint"] not in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "bigscience/bloom-560m",  "bert-base-uncased", "zhihan1996/DNA_bert_6"]:
  raise ValueError("Default Checkpoint Not Found")
if config["train_dataset"] not in ["11M", "100k", "30k", "3k"]:
  raise ValueError("Train Dataset Not Found")
if config["test_dataset"] not in ["11M", "100k", "30k", "3k"]:
  raise ValueError("Test Dataset Not Found")
if config["dataset_version"] not in ["small", "normal"]:
  raise ValueError("Dataset Version Should be Small or Normal")

if config["train_dataset"] == "11M":
  train_df = pd.read_csv("datasets/ExInSeqs_11M.csv", keep_default_na=False)
elif config["train_dataset"] == "100k":
  if config["dataset_version"] == "small":
    train_df = pd.read_csv("datasets/ExInSeqs_100k_small.csv", keep_default_na=False)
  else:
    train_df = pd.read_csv("datasets/ExInSeqs_100k.csv", keep_default_na=False)
elif config["train_dataset"] == "30k":
  if config["dataset_version"] == "small":
    train_df = pd.read_csv("datasets/ExInSeqs_30k_small.csv", keep_default_na=False)
  else:
    train_df = pd.read_csv("datasets/ExInSeqs_30k.csv", keep_default_na=False)
elif config["train_dataset"] == "3k":
  if config["dataset_version"] == "small":
    train_df = pd.read_csv("datasets/ExInSeqs_3k_small.csv", keep_default_na=False)
  else:
    train_df = pd.read_csv("datasets/ExInSeqs_3k.csv", keep_default_na=False)

if config["test_dataset"] == "11M":
  test_df = pd.read_csv("datasets/ExInSeqs_11M.csv", keep_default_na=False)
elif config["test_dataset"] == "100k":
  if config["dataset_version"] == "small":
    test_df = pd.read_csv("datasets/ExInSeqs_100k_small.csv", keep_default_na=False)
  else:
    test_df = pd.read_csv("datasets/ExInSeqs_100k.csv", keep_default_na=False)
elif config["test_dataset"] == "30k":
  if config["dataset_version"] == "small":
    test_df = pd.read_csv("datasets/ExInSeqs_30k_small.csv", keep_default_na=False)
  else:
    test_df = pd.read_csv("datasets/ExInSeqs_30k.csv", keep_default_na=False)
elif config["test_dataset"] == "3k":
  if config["dataset_version"] == "small":
    test_df = pd.read_csv("datasets/ExInSeqs_3k_small.csv", keep_default_na=False)
  else:
    test_df = pd.read_csv("datasets/ExInSeqs_3k.csv", keep_default_na=False)

train_sequence = train_df.iloc[:, 0].tolist()
train_label = train_df.iloc[:, 1].tolist()
train_organism = train_df.iloc[:, 2].tolist()
train_gene = train_df.iloc[:, 3].tolist()
train_flank_before = train_df.iloc[:, 4].tolist()
train_flank_after = train_df.iloc[:, 5].tolist()

test_sequence = test_df.iloc[:, 0].tolist()
test_label = test_df.iloc[:, 1].tolist()
test_organism = test_df.iloc[:, 2].tolist()
test_gene = test_df.iloc[:, 3].tolist()
test_flank_before = test_df.iloc[:, 4].tolist()
test_flank_after = test_df.iloc[:, 5].tolist()

sequence_len = 512
flanks_len = 25
if config["dataset_version"] == "small":
  sequence_len = 128
  flanks_len = 10

if config["checkpoint"] in  ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "bigscience/bloom-560m"]:
  model = SpliceGPT(checkpoint=config["checkpoint"], seed=config["seed"], alias=config["name"])
  model.add_train_data({
    "sequence": train_sequence,
    "label": train_label,
    "organism": train_organism,
    "gene": train_gene,
    "flank_before": train_flank_before,
    "flank_after": train_flank_after
  }, batch_size=config["batch_size"], sequence_len=sequence_len, flanks_len=flanks_len, train_percentage=config["train_percentage"], feat_hide_prob=config["hide_prob"])
  model.add_test_data({
    "sequence": test_sequence,
    "label": test_label,
    "organism": test_organism,
    "gene": test_gene,
    "flank_before": test_flank_before,
    "flank_after": test_flank_after
  }, batch_size=config["batch_size"], sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=config["hide_prob"])
  model.train(lr=config["lr"], epochs=config["epochs"], save_at_end=True, evaluation=False, save_freq=1)
  model.evaluate()

if config["checkpoint"] == "bert-base-uncased":
  model = SpliceBERT(checkpoint=config["checkpoint"], seed=config["seed"], alias=config["name"])
  model.add_train_data({
    "sequence": train_sequence,
    "label": train_label,
    "organism": train_organism,
    "gene": train_gene,
    "flank_before": train_flank_before,
    "flank_after": train_flank_after
  }, batch_size=config["batch_size"], sequence_len=sequence_len, flanks_len=flanks_len, train_percentage=config["train_percentage"], feat_hide_prob=config["hide_prob"])
  model.add_test_data({
    "sequence": test_sequence,
    "label": test_label,
    "organism": test_organism,
    "gene": test_gene,
    "flank_before": test_flank_before,
    "flank_after": test_flank_after
  }, batch_size=config["batch_size"], sequence_len=sequence_len, flanks_len=flanks_len, feat_hide_prob=config["hide_prob"])
  model.train(lr=config["lr"], epochs=config["epochs"], save_at_end=True, evaluation=False, save_freq=1)
  model.evaluate()

if config["checkpoint"] == "zhihan1996/DNA_bert_6":
  model = SpliceDNABERT(checkpoint=config["checkpoint"], seed=config["seed"], alias=config["name"])
  model.add_train_data({
    "sequence": train_sequence,
    "label": train_label,
    "organism": train_organism,
    "gene": train_gene,
    "flank_before": train_flank_before,
    "flank_after": train_flank_after
  }, batch_size=config["batch_size"], sequence_len=sequence_len, train_percentage=config["train_percentage"])
  model.add_test_data({
    "sequence": test_sequence,
    "label": test_label,
    "organism": test_organism,
    "gene": test_gene,
    "flank_before": test_flank_before,
    "flank_after": test_flank_after
  }, batch_size=config["batch_size"], sequence_len=sequence_len)
  model.train(lr=config["lr"], epochs=config["epochs"], save_at_end=True, evaluation=False, save_freq=1)
  model.evaluate()