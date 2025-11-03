from huggingface_hub import snapshot_download

folder = "src/storage/models/base/"

download_info = [
  {
    "model_id": "gpt2",
    "local_dir": folder + "gpt2"
  },
  {
    "model_id": "bert-base-uncased",
    "local_dir": folder + "bert"
  },
  {
    "model_id": "zhihan1996/DNABERT-2-117M",
    "local_dir": folder + "dnabert2"
  },
  {
    "model_id": "google-t5/t5-base",
    "local_dir": folder + "t5"
  }
]

for model_info in download_info:
  snapshot_download(
    repo_id=model_info["model_id"],
    local_dir=model_info["local_dir"]
  )