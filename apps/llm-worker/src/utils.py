import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from config import STORAGE_DIR
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def load_model(
	model_name: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
	checkpoint_path = os.path.join(STORAGE_DIR, "models", model_name)

	model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
	tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

	return model, tokenizer

def save_model(
	model_name: str,
	model: PreTrainedModel,
	tokenizer: PreTrainedTokenizerBase,
	history: dict | None
) -> None:
	output_path = os.path.join(STORAGE_DIR, "models", model_name)
	model.save_pretrained(output_path)
	tokenizer.save_pretrained(output_path)

	if history:
		df = pd.DataFrame(history)
		now = datetime.now().strftime("%Y%m%d-%H%M%S")
		df.to_csv(f"{output_path}/history-{now}.csv", index=False)

def set_seed(seed) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)

	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
	os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"