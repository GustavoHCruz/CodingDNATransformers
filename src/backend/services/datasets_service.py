from typing import Literal

from schemas.datasets_schema import CreationSettings

from backend.services.decorators import handle_exceptions
from src.backend.etl.genbank import splicing_sites_extraction as ss_genbank
from src.backend.etl.genbank import splicing_sites_extraction as ss_gencode


@handle_exceptions
def process_raw(settings: CreationSettings) -> Literal["Created"]:
	if settings["gencode"]:
		ss_genbank()
	if settings["genbank"]:
		ss_gencode()
	return "Created"