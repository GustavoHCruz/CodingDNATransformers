from backend.datasets.genbank.extraction import \
    splicing_sites_extraction as ss_genbank
from backend.datasets.gencode.extraction import \
    splicing_sites_extraction as ss_gencode
from backend.services.decorators import handle_exceptions


@handle_exceptions
def process_raw(settings: any) -> any:
	if settings["gencode"]:
		ss_genbank()
	return