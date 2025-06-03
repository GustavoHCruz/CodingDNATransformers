from typing import List, Optional

from models.base_model import ApproachEnum, BaseModel
from sqlmodel import Field, Relationship


class ModelHistory(BaseModel, table=True):
    parent_model_id: Optional[int] = Field(foreign_key="modelhistory.id", index=True)
    name: str
    approach: ApproachEnum
    model_name: str
    path: str
    seed: int
    epochs: int
    lr: float
    batch_size: int
    acc: float
    duration_seconds: int
    hide_prob: Optional[float] = None

    parent_model: Optional["ModelHistory"] = Relationship(
        back_populates="child_models",
        sa_relationship_kwargs={"remote_side": "ModelHistory.id"}
    )
    child_models: List["ModelHistory"] = Relationship(
        back_populates="parent_model"
    )

    train_history: List["TrainHistory"] = Relationship(back_populates="model")
    eval_history: List["EvalHistory"] = Relationship(back_populates="model")
