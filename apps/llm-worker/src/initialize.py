import os
from transformers import AutoModel, AutoTokenizer
from config import STORAGE_DIR

gpt_models = ['gpt2', 'gpt2-medium'];

models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'bert-base-uncased', 'zhihan1996/DNA_bert_6', 't5-base']

def preload_models():
  for model in models:
    AutoTokenizer.from_pretrained(model, cache_dir=os.path.join(STORAGE_DIR, "models"))
    AutoModel.from_pretrained(model, cache_dir=os.path.join(STORAGE_DIR, "models"))

if __name__ == "__main__":
  preload_models()