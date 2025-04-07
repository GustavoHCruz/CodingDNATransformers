import json


def read_supported_models():
	with open("configs/supported_models.json", "r") as file:
		supported_models = json.load(file)

		return {
			**supported_models,
			"all": supported_models["gpt"]+supported_models["bert"]+supported_models["dnabert"]+supported_models["t5"]
		}
	
def get_supported_models(approach):
	supported_models = read_supported_models()
	
	if approach == "ExInSeqs":
		return supported_models["gpt"]+supported_models["bert"]+supported_models["dnabert"]
	elif approach == "RebuildSeqs":
		return supported_models["gpt"]+supported_models["t5"]
	elif approach == "SWExInSeqs":
		return supported_models["bert"]+supported_models["dnabert"]
	else:
		return supported_models["t5"]
	
def read_experiment_configs(approach):
	approachs = {
		"ExInSeqs": dict,
		"RebuildSeqs": dict,
		"SWExInSeqs": dict,
		"ProteinSeqs": dict,
	}

	with open("configs/config.json", "r") as file:
		data = json.load(file)
	
	if approach not in approachs.keys():
		raise ValueError(f"Key {approach} not found on experiments.")

	supported_models = get_supported_models()

	data = data[approach]

	expected_types = {
		"model_name": str,
		"checkpoint_default": bool,
		"checkpoint_base": str,
		"checkpoint_to_load": str or None,
		"dataset_version": "small" or "normal",
		"dataset_source": "genbank" or "gencode",
		"train_dataset": str or None,
		"test_dataset": str or None,
		"batch_size": int,
		"lr": float,
		"epochs": int,
		"seed": int or None,
		"log_level": str
	}

	if approach == "ExInSeqs":
		expected_types.update({
			"hide_prob": float,
		})

		for key, expected_type in expected_types.items():
			if key not in data:
				raise ValueError(f"Missing key: {key}")
			
			if key == "checkpoint_base":
				if data[key] not in supported_models:
					raise TypeError(f"checkpoint_base doesn't match any of the supported models.")
			
			if not isinstance(data[key], expected_type):
				raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
			
			response = {
				"model_name": data["model_name"],
				"checkpoint_default": bool(data["checkpoint_default"]),
				"checkpoint_base": data["checkpoint_base"],
				"checkpoint_to_load": data["checkpoint_to_load"] if data["checkpoint_to_load"] != "None" else None,
				"dataset_version": data["dataset_version"],
				"dataset_source": data["dataset_source"],
				"train_dataset": data["train_dataset"] if data["train_dataset"] != "None" else None,
				"test_dataset": data["test_dataset"] if data["test_dataset"] != "None" else None,
				"batch_size": int(data["batch_size"]),
				"lr": float(data["lr"]),
				"epochs": int(data["epochs"]),
				"seed": int(data["seed"]),
				"log_level": str(data["log_level"]),
				"supported_models": supported_models
			}

			if approach == "ExInSeqs":
				response.update({"hide_prob": float(data["hide_prob"])})

			return response
	
def read_datasets_configs(approach, source):
	approachs = ["ExInSeqs", "RebuildSeqs", "SWExInSeqs"]
	sources = ["genbank", "gencode"]

	with open("configs/datasets.json", "r") as file:
		data = json.load(file)

	if approach not in approachs:
		raise ValueError(f"Key {approach} not found on approachs.")

	data = data[approach]

	if source not in sources:
		raise ValueError(f"Key {source} not found on sources.")

	data = data[source]

	expected_versions = {
		"default": dict,
		"small": dict
	}

	for key, expected_version in expected_versions.item():
		if key not in data:
			raise ValueError(f"missing key: {key}")
		if not isinstance(data[key], expected_version):
			raise TypeError(f"Incorrect type for '{key}': Expected {expected_version.__name__}, but got {type(data[key]).__name__}")

	configs = ["sequence_length", "sizes"]
	if approach == "ExInSeqs" or approach == "SWExInSeqs":
		configs.append("flanks")

	for key in configs:
		if key not in data:
			raise ValueError(f"missing key: {key}")

	for item in data["sizes"]:
		if not isinstance(item, dict):
				raise TypeError(f"Each item in 'sizes' must be a dict, but got {type(item).__name__}")
		if "name" not in item or "length" not in item:
			raise ValueError("Each item in 'sizes' must contain 'name' and 'length' keys")
		if not isinstance(item["name"], str):
			raise TypeError(f"Incorrect type for 'name': Expected str, but got {type(item['name']).__name__}")
		if not isinstance(item["length"], int):
			raise TypeError(f"Incorrect type for 'len': Expected int, but got {type(item['length']).__name__}")

	return data
