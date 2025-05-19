import threading
from datetime import datetime
from typing import Callable, Literal, Optional

from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from etl.gencode import (exin_classifier_gc, exin_translator_gc,
                         protein_translator_gc, sliding_window_tagger_gc)
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



def launch_background_task(
	extractor_func: Callable,
	annotations_path: str,
	approach: ApproachEnum,
	origin: OriginEnum,
	fasta_path: Optional[str] = None
) -> int:
	batch_size = 250
	total_records, progress_type = initial_configs(annotations_path, approach)

	parent_id = create_parent_dataset(ParentDataset(
		name=f"{approach.value}-{datetime.now()}",
		approach=approach,
		origin=origin
	)).id

	task_id = create_progress(
		progress_type,
		f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}"
	).id

	def background_task() -> None:
		if fasta_path:
			records_generator = extractor_func(fasta_path, annotations_path, parent_id)
		else:
			records_generator = extractor_func(annotations_path, parent_id)

		accepted_total, raw_total = bulk_create_parent_record_from_generator(
			records_generator, batch_size, task_id, total_records
		)
		update_parent_dataset_record_counter(parent_id, accepted_total)

		if not total_records:
			save_file(annotations_path, approach, raw_total)

	threading.Thread(target=background_task).start()
	return task_id

@handle_exceptions
def process_raw(settings: CreationSettings) -> CreationSettingsResponse:
	response = CreationSettingsResponse()
		
	if settings.genbank:
		origin = OriginEnum.genbank
		genbank_file_path = "data/raw/genbank/file1.gb"

		if settings.genbank.ExInClassifier:
			task_id = launch_background_task(
				extractor_func=exin_classifier_gb,
				annotations_path=genbank_file_path,
				approach=ApproachEnum.exin_classifier,
				origin=origin
			)
			response.genbank.ExInClassifier = task_id
		
		if settings.genbank.ExInTranslator:
			task_id = launch_background_task(
				extractor_func=exin_translator_gb,
				annotations_path=genbank_file_path,
				approach=ApproachEnum.exin_translator,
				origin=origin
			)
			response.genbank.ExInTranslator = task_id
		
		if settings.genbank.SlidingWindowTagger:
			task_id = launch_background_task(
				extractor_func=sliding_window_tagger_gb,
				annotations_path=genbank_file_path,
				approach=ApproachEnum.sliding_window_extraction,
				origin=origin
			)
			response.genbank.SlidingWindowTagger = task_id
		
		if settings.genbank.ProteinTranslator:
			task_id = launch_background_task(
				extractor_func=protein_translator_gb,
				annotations_path=genbank_file_path,
				approach=ApproachEnum.protein_translator,
				origin=origin
			)
			response.genbank.ProteinTranslator = task_id

	if settings.gencode:
		origin = OriginEnum.gencode
		gencode_fasta_path = "data/raw/gencode/file1.fa"
		gencode_annotations_path = "data/raw/gencode/file1.gtf"
		
		if settings.gencode.ExInClassifier:
			task_id = launch_background_task(
				extractor_func=exin_classifier_gc,
				annotations_path=gencode_annotations_path,
				approach=ApproachEnum.exin_classifier,
				origin=origin,
				fasta_path=gencode_fasta_path
			)
			response.gencode.ExInClassifier = task_id
		
		if settings.gencode.ExInTranslator:
			task_id = launch_background_task(
				extractor_func=exin_translator_gc,
				annotations_path=gencode_annotations_path,
				approach=ApproachEnum.exin_translator,
				origin=origin,
				fasta_path=gencode_fasta_path
			)
			response.gencode.ExInTranslator = task_id
		
		if settings.gencode.SlidingWindowTagger:
			task_id = launch_background_task(
				extractor_func=sliding_window_tagger_gc,
				annotations_path=gencode_annotations_path,
				approach=ApproachEnum.sliding_window_extraction,
				origin=origin,
				fasta_path=gencode_fasta_path
			)
			response.gencode.SlidingWindowTagger = task_id
		
		if settings.gencode.ProteinTranslator:
			task_id = launch_background_task(
				extractor_func=protein_translator_gc,
				annotations_path=gencode_annotations_path,
				approach=ApproachEnum.protein_translator,
				origin=origin,
				fasta_path=gencode_fasta_path
			)
			response.gencode.ProteinTranslator = task_id

	return response