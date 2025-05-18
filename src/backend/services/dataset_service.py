import threading
from datetime import datetime
from typing import Literal

from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from etl.gencode import exin_classifier_gc, exin_translator_gc, sliding_window_tagger_gc
from models.parent_dataset_model import ApproachEnum, OriginEnum, ParentDataset
from models.progress_tracker_model import ProgressTypeEnum
from schemas.datasets_schema import CreationSettings, CreationSettingsResponse
from services.decorators import handle_exceptions
from services.parent_dataset_service import (
    create_parent_dataset, update_parent_dataset_record_counter)
from services.parent_record_service import \
    bulk_create_parent_record_from_generator
from services.progress_tracker_service import create_progress
from services.raw_file_info_service import (get_by_file_name_and_approach,
                                            save_file)


def initial_configs(path: str, approach: ApproachEnum) -> tuple[int, Literal[ProgressTypeEnum.percentage, ProgressTypeEnum.counter]]:
	raw_file_info = get_by_file_name_and_approach(path, approach)

	total_records = None
	progress_type = ProgressTypeEnum.counter
	if raw_file_info:
		total_records = raw_file_info.total_records
		progress_type = ProgressTypeEnum.percentage

	return total_records, progress_type

@handle_exceptions
def process_raw(settings: CreationSettings) -> CreationSettingsResponse:
	response = CreationSettingsResponse()
	batch_size = 250
		
	if settings.genbank:
		origin = OriginEnum.genbank
		genbank_file_path = "data/raw/genbank/file1.gb"

		if settings.genbank.ExInClassifier:
			approach = ApproachEnum.exin_classifier
			total_records, progress_type = initial_configs(genbank_file_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id
			response.genbank.ExInClassifier = task_id
			def background_exin_classifier_gb() -> None:
				records_generator = exin_classifier_gb(genbank_file_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_generator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(genbank_file_path, approach, raw_total)

			threading.Thread(target=background_exin_classifier_gb).start()
		
		if settings.genbank.ExInTranslator:
			approach = ApproachEnum.exin_translator
			total_records, progress_type = initial_configs(genbank_file_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id
			response.genbank.ExInTranslator = task_id

			def background_exin_translator_gb() -> None:
				records_generator = exin_translator_gb(genbank_file_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_generator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(genbank_file_path, approach, raw_total)
			
			threading.Thread(target=background_exin_translator_gb).start()
		
		if settings.genbank.SlidingWindowTagger:
			approach = ApproachEnum.sliding_window_extraction
			total_records, progress_type = initial_configs(genbank_file_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id
			response.genbank.SlidingWindowTagger = task_id

			def background_sliding_window_tagger_gb() -> None:
				records_generator = sliding_window_tagger_gb(genbank_file_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_generator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(genbank_file_path, approach, raw_total)
			
			threading.Thread(target=background_sliding_window_tagger_gb).start()
		
		if settings.genbank.ProteinTranslator:
			approach = ApproachEnum.protein_translator
			total_records, progress_type = initial_configs(genbank_file_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id
			response.genbank.ProteinTranslator = task_id

			def background_protein_translator_gb() -> None:
				records_generator = protein_translator_gb(genbank_file_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_generator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(genbank_file_path, approach, raw_total)

			threading.Thread(target=background_protein_translator_gb).start()

	if settings.gencode:
		origin = OriginEnum.gencode
		gencode_fasta_path = "data/raw/gencode/file1.fa"
		gencode_annotations_path = "data/raw/gencode/file1.gtf"
		
		if settings.gencode.ExInClassifier:
			approach = ApproachEnum.exin_classifier
			total_records, progress_type = initial_configs(gencode_annotations_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id		
			response.gencode.ExInClassifier = task_id

			def background_exin_classifier_gc() -> None:
				records_renerator = exin_classifier_gc(gencode_fasta_path, gencode_annotations_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_renerator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(gencode_annotations_path, approach, raw_total)

			threading.Thread(target=background_exin_classifier_gc).start()
		
		if settings.gencode.ExInTranslator:
			approach = ApproachEnum.exin_translator
			total_records, progress_type = initial_configs(gencode_annotations_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id		
			response.gencode.ExInTranslator = task_id

			def background_exin_translator_gc() -> None:
				records_renerator = exin_translator_gc(gencode_fasta_path, gencode_annotations_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_renerator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(gencode_annotations_path, approach, raw_total)

			threading.Thread(target=background_exin_translator_gc).start()
		
		if settings.gencode.SlidingWindowTagger:
			approach = ApproachEnum.sliding_window_extraction
			total_records, progress_type = initial_configs(gencode_annotations_path, approach)

			parent_id = create_parent_dataset(ParentDataset(
				name=f"{approach}-{datetime.now()}",
				approach=approach,
				origin=origin
			)).id

			task_id = create_progress(progress_type, f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}").id		
			response.gencode.SlidingWindowTagger = task_id

			def background_sliding_window_tagger_gc() -> None:
				records_renerator = sliding_window_tagger_gc(gencode_fasta_path, gencode_annotations_path, parent_id)

				accepted_total, raw_total = bulk_create_parent_record_from_generator(records_renerator, batch_size, task_id, total_records)

				update_parent_dataset_record_counter(parent_id, accepted_total)

				if not total_records:
					save_file(gencode_annotations_path, approach, raw_total)

			threading.Thread(target=background_sliding_window_tagger_gc).start()

	return response