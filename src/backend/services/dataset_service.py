import threading
from datetime import datetime

from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from models.parent_dataset_model import ApproachEnum, OriginEnum, ParentDataset
from models.progress_tracker_model import ProgressTypeEnum
from schemas.datasets_schema import CreationSettings, CreationSettingsResponse
from services.decorators import handle_exceptions
from services.parent_dataset_service import (
    create_parent_dataset, update_parent_dataset_record_counter)
from services.parent_record_service import \
    bulk_create_parent_record_from_generator
from services.progress_tracker_service import create_progress
from services.raw_file_info_service import get_by_file_name


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
			progress_tracker = create_progress(progress_type)
			response.genbank.ExInClassifier = progress_tracker.id

			parent_dataset = ParentDataset(
				name=f"ExInClassifier-{datetime.now()}",
				approach=ApproachEnum.exin_classifier,
				origin=OriginEnum.genbank
			)

			parent_dataset = create_parent_dataset(parent_dataset)

			def background_exin_classifier_gb() -> None:
				records_generator = exin_classifier_gb("data/raw/genbank/file1.gb", total_records, new_reading, progress_tracker.id, parent_dataset.id)

				total = bulk_create_parent_record_from_generator(records_generator)

				update_parent_dataset_record_counter(parent_dataset.id, total)

			threading.Thread(target=background_exin_classifier_gb).start()
		
		if settings.genbank.ExInTranslator:
			progress_tracker = create_progress(progress_type)
			response.genbank.ExInTranslator = progress_tracker.id

			parent_dataset = ParentDataset(
				name=f"ExInTranslator-{datetime.now()}",
				approach=ApproachEnum.exin_translator,
				origin=OriginEnum.genbank
			)

			parent_dataset = create_parent_dataset(parent_dataset)

			def background_exin_translator_gb() -> None:
				records_generator = exin_translator_gb("data/raw/genbank/file1.gb", total_records, new_reading, progress_tracker.id, parent_dataset.id)

				total = bulk_create_parent_record_from_generator(records_generator)

				update_parent_dataset_record_counter(parent_dataset.id, total)
			
			threading.Thread(target=background_exin_translator_gb).start()
		
		if settings.genbank.SlidingWindowTagger:
			progress_tracker = create_progress(progress_type)
			response.genbank.SlidingWindowTagger = progress_tracker.id

			parent_dataset = ParentDataset(
				name=f"SlidingWindowTagger-{datetime.now()}",
				approach=ApproachEnum.sliding_window_extraction,
				origin=OriginEnum.genbank
			)

			parent_dataset = create_parent_dataset(parent_dataset)

			def background_sliding_window_tagger_gb() -> None:
				records_generator = sliding_window_tagger_gb("data/raw/genbank/file1.gb", total_records, new_reading, progress_tracker.id, parent_dataset.id)

				total = bulk_create_parent_record_from_generator(records_generator)

				update_parent_dataset_record_counter(parent_dataset.id, total)
			
			threading.Thread(target=background_sliding_window_tagger_gb).start()
		
		if settings.genbank.ProteinTranslator:
			progress_tracker = create_progress(progress_type)
			response.genbank.ProteinTranslator = progress_tracker.id

			parent_dataset = ParentDataset(
				name=f"ProteinTranslator-{datetime.now()}",
				approach=ApproachEnum.protein_translator,
				origin=OriginEnum.genbank
			)

			parent_dataset = create_parent_dataset(parent_dataset)

			def background_protein_translator_gb() -> None:
				records_generator = protein_translator_gb("data/raw/genbank/file1.gb", total_records, new_reading, progress_tracker.id, parent_dataset.id)

				total = bulk_create_parent_record_from_generator(records_generator)

				update_parent_dataset_record_counter(parent_dataset.id, total)
			
			threading.Thread(target=background_protein_translator_gb).start()

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