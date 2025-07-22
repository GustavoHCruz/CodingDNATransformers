import sys

import typer
from config import SHARED_DIR, STORAGE_DIR
from interfaces import ApproachEnum, ModelEnum
from llms.protein_translator.gpt import create_model as tp_gpt_create
from llms.protein_translator.gpt import train_model as tp_gpt_train

app = typer.Typer(help="CLI to train, evaluate and predict Splicing Sites using LLMs.")

sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(STORAGE_DIR))

load_dotenv()
redis = RedisService()

@app.command()
def create(
  approach: ApproachEnum,
  model: ModelEnum,
  uuid: str,
  checkpoint: str,
  name: str
) -> None:
  if approach == ApproachEnum.PROTEINTRANSLATOR:
    if model == ModelEnum.GPT:
      tp_gpt_create(
        uuid=uuid,
        checkpoint=checkpoint,
        name=name
      )

@app.command()
def train(
  approach: ApproachEnum,
  model: ModelEnum,
  uuid: str,
  name: str,
  seed: int,
  data_length: int,
  epochs: int,
  batch_size: int,
  lr: float,
  gradient_accumulation: int,
  warmup_ratio: float,
) -> None:
  if approach == ApproachEnum.PROTEINTRANSLATOR:
    if model == ModelEnum.GPT:
      tp_gpt_train(
        uuid=uuid,
        name=name,
        seed=seed,
        data_length=data_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        gradient_accumulation=gradient_accumulation,
        warmup_ratio=warmup_ratio
      )

@app.command()
def evaluate(
  approach: ApproachEnum,
  model: ModelEnum,
  args: str
) -> None:
  ...

@app.command()
def predict(
  approach: ApproachEnum,
  model: ModelEnum,
  args: str
) -> None:
  ...

if __name__ == "__main__":
  app()
