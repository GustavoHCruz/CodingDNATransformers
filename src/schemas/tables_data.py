from typing import Optional, TypedDict


class CDSSequence(TypedDict):
	sequence: str
	start: int
	end: int
	gene: str

class ExInSequence(TypedDict):
	sequence: str
	type: str
	start: int
	end: int
	gene: str
	strand: Optional[int]
	before: str
	after: str

class DNASequence(TypedDict):
	sequence: str
	accession: str
	organism: str
	cds: list[CDSSequence]
	exin: list[ExInSequence]
	exin: list[ExInSequence]
