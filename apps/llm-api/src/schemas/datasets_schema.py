from typing import Optional

from models.base_model import ApproachEnum
from models.parent_dataset_model import OriginEnum
from pydantic import BaseModel, Field


class Approachs(BaseModel):
	ExInClassifier: bool = False
	ExInTranslator: bool = False
	SlidingWindowTagger: bool = False
	ProteinTranslator: bool = False

class CreationSettings(BaseModel):
	genbank: Optional[Approachs] = None
	gencode: Optional[Approachs] = None

class ApproachsResponse(BaseModel):
	ExInClassifier: Optional[int] = None
	ExInTranslator: Optional[int] = None
	SlidingWindowTagger: Optional[int] = None
	ProteinTranslator: Optional[int] = None

class CreationSettingsResponse(BaseModel):
	genbank: ApproachsResponse = ApproachsResponse()
	gencode: ApproachsResponse = ApproachsResponse()

class ChildDatasetSettings(BaseModel):
	name: str
	size: int

class ProcessedDatasetCreation(BaseModel):
  batch_name: Optional[str] = None
  approach: ApproachEnum
  datasets: list[ChildDatasetSettings] = Field(default_factory=list)
  seed: Optional[int] = 1234

class ProcessedDatasetCreationResponse(BaseModel):
	name: str
	task_id: int
	