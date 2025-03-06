import pandas as pd

from classes.ExInSeqs_BERT import ExInSeqsBERT
from classes.ExInSeqs_DNABERT import ExInSeqsDNABERT
from classes.ExInSeqs_GPT import ExInSeqsGPT
from funcs.config_reading import read_datasets_configs, read_experiment_configs

config = read_experiment_configs("ExInSeqs")
datasets_config = read_datasets_configs("ExInSeqs")

dataset_names = [i["name"] for i in datasets_config["sizes"]]

if config["checkpoint_base"] not in config["supported_models"]["all"]:
  raise ValueError("Default Checkpoint Not Found")
if config["train_dataset"] not in dataset_names:
  raise ValueError("Train Dataset Not Found")
if config["test_dataset"] not in dataset_names:
  raise ValueError("Test Dataset Not Found")
if config["dataset_version"] not in ["small", "normal"]:
  raise ValueError("Dataset Version Should be Small or Normal")

train_df_path = f"datasets/{config["train_dataset"]}"
test_df_path = f"datasets/{config["test_dataset"]}"
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

sequence_len = datasets_config["version"]["normal"]["sequence_len"]
flanks_len = datasets_config["version"]["normal"]["flanks"]
if config["dataset_version"] == "small":
  sequence_len = datasets_config["version"]["small"]["sequence_len"]
  flanks_len = datasets_config["version"]["small"]["flanks"]

model_to_use = config["checkpoint_base"]
if not config["checkpoint_default"]:
  model_to_use = config["checkpoint_to_load"]

if config["checkpoint_base"] in config["supported_models"]["gpt"]:
  model = ExInSeqsGPT(model_to_use, seed=config["seed"], alias=config["model_name"], log_level=config["log_level"])
elif config["checkpoint_base"] == "bert-base-uncased":
  model = ExInSeqsGPT(model_to_use, seed=config["seed"], alias=config["model_name"], log_level=config["log_level"])
else:
  model = ExInSeqsDNABERT(model_to_use, seed=config["seed"], alias=config["model_name"], log_level=config["log_level"])

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

model.train(lr=config["lr"], epochs=config["epochs"], save_at_end=True, evaluation=False, save_freq=0)
model.evaluate()
