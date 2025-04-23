from typing import Optional

from pydantic import BaseModel


class Approachs(BaseModel):
	ExInClassifier: bool = False
	ExInTranslator: bool = False
	ProteinTranslator: bool = False
	SlidingWindowTagger: bool = False

class CreationSettings(BaseModel):
	genbank: Optional[Approachs] = None
	gencode: Optional[Approachs] = None

class ApproachsResponse(BaseModel):
	ExInClassifier: Optional[int] = None
	ExInTranslator: Optional[int] = None
	ProteinTranslator: Optional[int] = None
	SlidingWindowTagger: Optional[int] = None

class CreationSettingsResponse(BaseModel):
	genbank: ApproachsResponse = ApproachsResponse()
	gencode: ApproachsResponse = ApproachsResponse()