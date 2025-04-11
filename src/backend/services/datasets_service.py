from app.backend.models.datasets_model import DatasetsRaw
from app.backend.services.decorators import handle_exceptions
from pipelines.datasets.genbank.extraction import \
    splicing_sites_extraction as ss_genbank
from pipelines.datasets.gencode.extraction import \
    splicing_sites_extraction as ss_gencode


@handle_exceptions
def process_raw(settings: DatasetsRaw):
	if settings["gencode"]:
		ss_genbank()
	return