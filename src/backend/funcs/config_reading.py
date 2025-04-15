import json


def read_config_file():
	expected_attr = {
		"datasets_raw_dir": str,
		"datasets_dir": str,
		"cache_dir": str,
		"cache_file": str,
		"genbank_file": str,
		"gencode_file_annotations": str,
		"gencode_file_genome": str,
	}

	with open(read_config_file("config.json"), "r") as file:
		data = json.load(file)

	for key, expected_type in expected_attr.items():
		if key not in data:
			raise ValueError(f"Missing key: {key}")
			
		if not isinstance(data[key], expected_type):
			raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
			
		response = {}
		
		for key, expected_type in expected_attr.items():
			response.update({
				key: expected_type(data[key])
			})

		return response

def read_supported_models():
	with open(read_config_file("supported_models.json"), "r") as file:
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
