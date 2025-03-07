import json


def read_supported_models():
  with open("configs/supported_models.json", "r") as file:
    supported_models = json.load(file)

    return {
      **supported_models,
      "all": supported_models["gpt"]+supported_models["bert"]+supported_models["dnabert"],
    }

def read_experiment_configs(experiment):
  experiments = {
    "ExInSeqs": dict,
    "RebuildSeqs": dict,
  }

  with open("configs/config.json", "r") as file:
    data = json.load(file)

  supported_models = read_supported_models()
  
  if experiment not in experiments.keys():
    raise ValueError(f"Key {experiment} not found on experiments.")

  data = data[experiment]
  if experiment == "ExInSeqs":
    expected_types = {
      "model_name": str,
      "checkpoint_default": bool,
      "checkpoint_base": str,
      "checkpoint_to_load": str or None,
      "dataset_version": "small" or "normal",
      "train_dataset": str,
      "test_dataset": str,
      "train_percentage": float,
      "batch_size": int,
      "hide_prob": float,
      "lr": float,
      "epochs": int,
      "seed": int or None,
      "log_level": str
    }
    
    for key, expected_type in expected_types.items():
      if key not in data:
        raise ValueError(f"Missing key: {key}")
      
      if key == "checkpoint_base":
        if data[key] not in supported_models["all"]:
          raise TypeError(f"checkpoint_base doesn't match any of the supported models.")
      
      if not isinstance(data[key], expected_type):
        raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
      
      return {
        "model_name": data["model_name"],
        "checkpoint_default": bool(data["checkpoint_default"]),
        "checkpoint_base": data["checkpoint_base"],
        "checkpoint_to_load": data["checkpoint_to_load"] if data["checkpoint_to_load"] != "None" else None,
        "dataset_version": data["dataset_version"],
        "train_dataset": data["train_dataset"],
        "test_dataset": data["test_dataset"],
        "train_percentage": float(data["train_percentage"]),
        "batch_size": int(data["batch_size"]),
        "hide_prob": float(data["hide_prob"]),
        "lr": float(data["lr"]),
        "epochs": int(data["epochs"]),
        "seed": int(data["seed"]),
        "log_level": str(data["log_level"]),
        "supported_models": supported_models
      }
  elif experiment == "RebuildSeqs":
    expected_types = {
      "model_name": str,
      "checkpoint_default": bool,
      "checkpoint_base": str,
      "checkpoint_to_load": str or None,
      "dataset_version": "small" or "normal",
      "train_dataset": str,
      "test_dataset": str,
      "batch_size": int,
      "lr": float,
      "epochs": int,
      "seed": int or None,
      "log_level": str
    }
    
    for key, expected_type in expected_types.items():
      if key not in data:
        raise ValueError(f"Missing key: {key}")
      
      if key == "checkpoint_base":
        if data[key] not in supported_models["all"]:
          raise TypeError(f"checkpoint_base doesn't match any of the supported models.")
      
      if not isinstance(data[key], expected_type):
        raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
      
      return {
        "model_name": data["model_name"],
        "checkpoint_default": bool(data["checkpoint_default"]),
        "checkpoint_base": data["checkpoint_base"],
        "checkpoint_to_load": data["checkpoint_to_load"] if data["checkpoint_to_load"] != "None" else None,
        "dataset_version": data["dataset_version"],
        "train_dataset": data["train_dataset"],
        "test_dataset": data["test_dataset"],
        "batch_size": int(data["batch_size"]),
        "lr": float(data["lr"]),
        "epochs": int(data["epochs"]),
        "seed": int(data["seed"]),
        "log_level": str(data["log_level"]),
        "supported_models": supported_models["gpt"]
      }
  
def read_datasets_configs(dataset):
  datasets = {
    "ExInSeqs": dict,
    "RebuildSeqs": dict,
    "SWExInSeqs": dict
  }
  
  with open("configs/datasets.json", "r") as file:
    data = json.load(file)

  if dataset not in datasets.keys():
    raise ValueError(f"Key {dataset} not found on datasets.")

  data = data[dataset]
  if dataset == "ExInSeqs":
    expected_types = {
      "sizes": list,
      "version": dict
    }
    for key, expected_type in expected_types.items():
      if key not in data:
        raise ValueError(f"Missing key: {key}")
      if not isinstance(data[key], expected_type):
        raise TypeError(f"Incorrect type for '{key}': Expected {expected_type.__name__}, but got {type(data[key]).__name__}")

    for item in data["sizes"]:
      if not isinstance(item, dict):
        raise TypeError(f"Each item in 'sizes' must be a dict, but got {type(item).__name__}")
      if "name" not in item or "len" not in item:
        raise ValueError("Each item in 'sizes' must contain 'name' and 'len' keys")
      if not isinstance(item["name"], str):
        raise TypeError(f"Incorrect type for 'name': Expected str, but got {type(item['name']).__name__}")
      if not isinstance(item["len"], int):
        raise TypeError(f"Incorrect type for 'len': Expected int, but got {type(item['len']).__name__}")

    for version_key in ["small", "normal"]:
      if version_key not in data["version"]:
        raise ValueError(f"Missing key in 'version': {version_key}")
      version = data["version"][version_key]
      if not isinstance(version, dict):
        raise TypeError(f"Incorrect type for 'version[{version_key}]': Expected dict, but got {type(version).__name__}")
      if "sequence_len" not in version or "flanks" not in version:
        raise ValueError(f"'version[{version_key}]' must contain 'sequence_len' and 'flanks' keys")
      if not isinstance(version["sequence_len"], int):
        raise TypeError(f"Incorrect type for 'sequence_len' in 'version[{version_key}]': Expected int, but got {type(version['sequence_len']).__name__}")
  elif dataset == "RebuildSeqs":
    expected_types = {
      "sizes": list,
      "version": dict
    }
    for key, expected_type in expected_types.items():
      if key not in data:
        raise ValueError(f"Missing key: {key}")
      if not isinstance(data[key], expected_type):
        raise TypeError(f"Incorrect type for '{key}': Expected {expected_type.__name__}, but got {type(data[key]).__name__}")

    for item in data["sizes"]:
      if not isinstance(item, dict):
        raise TypeError(f"Each item in 'sizes' must be a dict, but got {type(item).__name__}")
      if "name" not in item or "len" not in item:
        raise ValueError("Each item in 'sizes' must contain 'name' and 'len' keys")
      if not isinstance(item["name"], str):
        raise TypeError(f"Incorrect type for 'name': Expected str, but got {type(item['name']).__name__}")
      if not isinstance(item["len"], int):
        raise TypeError(f"Incorrect type for 'len': Expected int, but got {type(item['len']).__name__}")

    for version_key in ["small", "normal"]:
      if version_key not in data["version"]:
        raise ValueError(f"Missing key in 'version': {version_key}")
      version = data["version"][version_key]
      if not isinstance(version, dict):
        raise TypeError(f"Incorrect type for 'version[{version_key}]': Expected dict, but got {type(version).__name__}")
      if "sequence_len" not in version:
        raise ValueError(f"'version[{version_key}]' must contain 'sequence_len' key")
      if not isinstance(version["sequence_len"], int):
        raise TypeError(f"Incorrect type for 'sequence_len' in 'version[{version_key}]': Expected int, but got {type(version['sequence_len']).__name__}")
  else: 
    expected_types = {
      "sizes": list,
    }
    for key, expected_type in expected_types.items():
      if key not in data:
        raise ValueError(f"Missing key: {key}")
      if not isinstance(data[key], expected_type):
        raise TypeError(f"Incorrect type for '{key}': Expected {expected_type.__name__}, but got {type(data[key]).__name__}")

    for item in data["sizes"]:
      if not isinstance(item, dict):
        raise TypeError(f"Each item in 'sizes' must be a dict, but got {type(item).__name__}")
      if "name" not in item or "len" not in item:
        raise ValueError("Each item in 'sizes' must contain 'name' and 'len' keys")
      if not isinstance(item["name"], str):
        raise TypeError(f"Incorrect type for 'name': Expected str, but got {type(item['name']).__name__}")
      if not isinstance(item["len"], int):
        raise TypeError(f"Incorrect type for 'len': Expected int, but got {type(item['len']).__name__}")

  return data
