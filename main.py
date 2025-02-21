import pandas as pd

from BERT_based import SpliceBERT, SpliceDNABERT
from GPT_based import SpliceGPT

gpt_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "bigscience/bloom-560m"]
all_models = gpt_models + ["bert-base-uncased", "zhihan1996/DNA_bert_6"]

config = {
  "name": "bert-001",
  "checkpoint_default": True,
  "checkpoint_base": "gpt2",
  "checkpoint_to_load": None,
  "dataset_version": "small",
  "train_dataset": "100k",
  "test_dataset": "30k",
  "train_percentage": 1.0,
  "batch_size": 32,
  "hide_prob": 0.4,
  "lr": 5e-5,
  "epochs": 3,
  "seed": 1234
}

if config["checkpoint_base"] not in all_models:
  raise ValueError("Default Checkpoint Not Found")
if config["train_dataset"] not in ["5M", "100k", "30k", "3k"]:
  raise ValueError("Train Dataset Not Found")
if config["test_dataset"] not in ["5M", "100k", "30k", "3k"]:
  raise ValueError("Test Dataset Not Found")
if config["dataset_version"] not in ["small", "normal"]:
  raise ValueError("Dataset Version Should be Small or Normal")

train_df_path = f"datasets/ExInSeqs_{config["train_dataset"]}"
test_df_path = f"datasets/ExInSeqs_{config["test_dataset"]}"
if config["dataset_version"] == "small":
  train_df_path += "_small.csv"
  test_df_path += "_small.csv"
else:
  train_df_path += ".csv"
  test_df_path += ".csv"

train_df = pd.read_csv(train_df_path, keep_default_na=False)
test_df = pd.read_csv(test_df_path, keep_default_na=False)

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

model_to_use = config["checkpoint_base"]
if not config["checkpoint_default"]:
  model_to_use = config["checkpoint_to_load"]

if config["checkpoint_base"] in gpt_models:
  model = SpliceGPT(model_to_use, seed=config["seed"], alias=config["name"])
elif config["checkpoint_base"] == "bert-base-uncased":
  model = SpliceBERT(model_to_use, seed=config["seed"], alias=config["name"])
else:
  model = SpliceDNABERT(model_to_use, seed=config["seed"], alias=config["name"])

data_config = {
  "flanks_len": flanks_len,
  "feat_hide_prob": config["hide_prob"],
}

model.add_train_data({
  "sequence": train_sequence,
  "label": train_label,
  "organism": train_organism,
  "gene": train_gene,
  "flank_before": train_flank_before,
  "flank_after": train_flank_after
}, batch_size=config["batch_size"], sequence_len=sequence_len, train_percentage=config["train_percentage"], data_config=data_config)

model.add_test_data({
  "sequence": test_sequence,
  "label": test_label,
  "organism": test_organism,
  "gene": test_gene,
  "flank_before": test_flank_before,
  "flank_after": test_flank_after
}, batch_size=config["batch_size"], sequence_len=sequence_len, data_config=data_config)

model.train(lr=config["lr"], epochs=config["epochs"], save_at_end=True, evaluation=False, save_freq=1)
model.evaluate()
