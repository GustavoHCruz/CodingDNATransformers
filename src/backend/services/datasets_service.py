import threading
from datetime import datetime

from database.db import get_session
from etl.genbank import exin_classifier as exin_classifier_gb
from models.datasets_model import (ApproachEnum, Datasets, DatasetTypeEnum,
                                   OriginEnum)
from models.progress_tracker_model import ProgressTypeEnum
from schemas.datasets_schema import CreationSettings, CreationSettingsResponse
from services.decorators import handle_exceptions
from services.exin_classifier_service import bulk_create
from services.progress_tracker_service import create_progress
from services.raw_file_info_service import get_by_file_name


@handle_exceptions
def create_dataset(data: Datasets) -> Datasets:
	with get_session() as session:
		session.add(data)
		session.commit()
		session.refresh(data)
		return data

@handle_exceptions
def process_raw(settings: CreationSettings) -> CreationSettingsResponse:
	response = CreationSettingsResponse()
		
	if settings.genbank:
		raw_file_info = get_by_file_name("data/raw/genbank/file1.gb")

		new_reading = True
		total_records = 0
		progress_type = ProgressTypeEnum.counter
		if raw_file_info:
			total_records = raw_file_info.total_records
			new_reading = False
			progress_type = ProgressTypeEnum.percentage

		if settings.genbank.ExInClassifier:
			progress_tracker = create_progress(type=progress_type)
			response.genbank.ExInClassifier = progress_tracker.id

			dataset_data = Datasets(
				approach=ApproachEnum.exin_classifier,
				name=f"ExInClassifier-{datetime.now()}",
				dataset_type=DatasetTypeEnum.base,
				origin=OriginEnum.genbank
			)
			dataset = create_dataset(dataset_data)

			def background_exin_classifier_gb() -> None:
				data = exin_classifier_gb("data/raw/genbank/file1.gb", total_records, new_reading, progress_tracker.id, dataset.id)

				bulk_create(data)

			threading.Thread(target=background_exin_classifier_gb).start()

	if settings.gencode:
		raw_file_info = get_by_file_name("data/raw/gencode/file1.fa")

		new_reading = True
		total_records = 0
		progress_type = ProgressTypeEnum.counter
		if raw_file_info:
			total_records = raw_file_info.total_records
			new_reading = False
			progress_type = ProgressTypeEnum.percentage
		
		if settings.gencode.ExInClassifier:
			progress_tracker = create_progress(type=progress_type)		
			response.gencode.ExInClassifier = progress_tracker.id

			# call ExInClassifier(progress_tracker.id) - gencode

			response.gencode.ExInClassifier = progress_tracker.id

	return response