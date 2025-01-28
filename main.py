import pandas as pd

from BERT_based import SpliceBERT, SpliceDNABERT
from genbank_dataset_extraction import splicing_sites_extraction
from GPT_based import SpliceGPT

config = {
  "name": "gpt2-001",
  "checkpoint_default": True,
  "checkpoint": "gpt2",
  "train_dataset": "100k",
  "train_percentage": 0.8,
  "test_dataset": "3k",
  "dataset_version": "small",
  "seed": 1234,
  "batch_size": 16,
  "hide_prob": 0.4
}

if config["checkpoint"] not in ["gpt2", "bert", "dnabert"]:
  raise ValueError("Default Checkpoint Not Found")
if config["train_dataset"] not in ["11M", "100k", "3k"]:
  raise ValueError("Train Dataset Not Found")
if config["test_dataset"] not in ["11M", "100k", "3k"]:
  raise ValueError("Test Dataset Not Found")

if config["train_dataset"] == "11M":
  train_df = pd.read_csv("datasets/ExInSeqs_11M.csv", keep_default_na=False)
elif config["train_dataset"] == "100k":
  if config["dataset_version"] == "small":
    train_df = pd.read_csv("datasets/ExInSeqs_100k_small.csv", keep_default_na=False)
  else:
    train_df = pd.read_csv("datasets/ExInSeqs_100k.csv", keep_default_na=False)
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

name = config["name"]
if config["checkpoint_default"]:
  name = config["checkpoint"]

if config["checkpoint"] == "gpt2":
  model = SpliceGPT(checkpoint=name, seed=config["seed"], alias=name)

if config["checkpoint"] == "bert":
  model = SpliceBERT(checkpoint=name, seed=config["seed"], alias=name)

if config["checkpoint"] == "dnabert":
  model = SpliceBERT(checkpoint=name, seed=config["seed"], alias=name)