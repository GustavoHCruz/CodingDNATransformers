from etl.genbank import splicing_sites_extraction as ss_genbank
from etl.genbank import splicing_sites_extraction as ss_gencode
from models.progress_model import ProgressTypeEnum
from schemas.datasets_schema import CreationSettings, CreationSettingsResponse
from services.decorators import handle_exceptions
from services.progress_tracker_service import create_progress
from services.raw_file_info_service import get_by_file_name


@handle_exceptions
def process_raw(settings: CreationSettings) -> CreationSettingsResponse:

	response = CreationSettingsResponse
	new_reading = True
	if raw_file_info:
		total_records = raw_file_info.total_records
		new_reading = False
		
	progress_type = ProgressTypeEnum.counter if new_reading else ProgressTypeEnum.percentage

	if settings.genbank:
		raw_file_info = get_by_file_name("data/raw/genbank/file1.gb")

		if settings.genbank.ExInClassifier:
			progress_tracker = create_progress(type=progress_type)
			response.genbank.ExInClassifier = progress_tracker.id

			# call ExInClassifier(progress_tracker.id) - Genbank


	if settings.gencode:
		raw_file_info = get_by_file_name("data/raw/gencode/file1.fa")
		
		if settings.gencode.ExInClassifier:
			progress_tracker = create_progress(type=progress_type)		
			response.gencode.ExInClassifier = progress_tracker.id

			# call ExInClassifier(progress_tracker.id) - gencode

	return response