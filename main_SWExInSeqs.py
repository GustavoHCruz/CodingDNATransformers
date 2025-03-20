import pandas as pd

from classes.SWExInSeqs_BERT import SWExInSeqsBERT
from funcs.config_reading import read_datasets_configs, read_experiment_configs

config = read_experiment_configs("SWExInSeqs")
datasets_config = read_datasets_configs("SWExInSeqs")

dataset_names = [i["name"] for i in datasets_config["sizes"]]

if config["checkpoint_base"] not in config["supported_models"]:
  raise ValueError("Default Checkpoint Not Found")
if config["train_dataset"] not in dataset_names:
  raise ValueError("Train Dataset Not Found")
if config["test_dataset"] not in dataset_names:
  raise ValueError("Test Dataset Not Found")

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
train_organism = train_df.iloc[:, 1].tolist()
train_label = train_df.iloc[:, 2].tolist()

test_sequence = test_df.iloc[:, 0].tolist()
test_organism = test_df.iloc[:, 1].tolist()
test_label = test_df.iloc[:, 2].tolist()

model_to_use = config["checkpoint_base"]
if not config["checkpoint_default"]:
  model_to_use = config["checkpoint_to_load"]

model = SWExInSeqsBERT(model_to_use, seed=config["seed"], alias=config["model_name"], log_level=config["log_level"], window_size=config["window_size"])

data_config = {
  "flank_size": config["flank_size"]
}

model.add_train_data({
  "sequence": train_sequence,
  "organism": train_organism,
  "labeled_sequence": train_label,
}, batch_size=config["batch_size"], data_config=data_config)

model.add_test_data({
  "sequence": test_sequence,
  "organism": test_organism,
  "labeled_sequence": test_label,
}, batch_size=config["batch_size"], data_config=data_config)

model.train(lr=config["lr"], epochs=config["epochs"], save_at_end=True, save_freq=0)
model.evaluate()
