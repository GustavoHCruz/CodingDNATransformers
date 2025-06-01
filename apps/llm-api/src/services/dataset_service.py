import random
import threading
from datetime import datetime
from typing import Callable, Literal, Optional

from config import config
from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from etl.gencode import (exin_classifier_gc, exin_translator_gc,
                         protein_translator_gc, sliding_window_tagger_gc)
from models.child_dataset_model import ChildDataset
from models.child_record_model import ChildRecord
from models.generation_batch_model import GenerationBatch
from models.parent_dataset_model import ApproachEnum, OriginEnum, ParentDataset
from models.progress_tracker_model import ProgressTypeEnum
from repositories.child_dataset_repo import create_child_dataset
from repositories.child_record_repo import bulk_create_child_records
from repositories.generation_batch_repo import create_generation_batch
from repositories.parent_dataset_repo import (
    create_parent_dataset, update_parent_dataset_record_counter)
from repositories.parent_record_repo import (
    bulk_create_parent_record_from_generator,
    get_parents_records_ids_by_approach)
from repositories.progress_tracker_repo import create_progress, finish_progress
from repositories.raw_file_info_repo import (get_by_file_name_and_approach,
                                             save_file)
from schemas.datasets_schema import (CreationSettings,
                                     CreationSettingsResponse,
                                     ProcessedDatasetCreation,
                                     ProcessedDatasetCreationResponse)
from services.progress_tracker_service import post_progress

from utils import batch_list


def initial_configs(path: str, approach: ApproachEnum) -> tuple[int, Literal[ProgressTypeEnum.percentage, ProgressTypeEnum.counter]]:
	raw_file_info = get_by_file_name_and_approach(path, approach)

	total_records = 0
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

	if not parent_id:
		raise ValueError("Could not retrieve Parent Dataset")

	task_id = create_progress(
		progress_type,
		f"origin:{origin.value}-approach:{approach.value}-parent:{parent_id}"
	).id

	if not task_id:
		raise ValueError("Could not retrieve Progress Tracker")

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

def validate_size(total_available: int, current_amount: int, requested_amount: int) -> bool:
	if total_available > (current_amount + requested_amount):
		return True
	
	return False

def background_child_dataset_creation(
		batch_generation_id: int,
		approach: ApproachEnum,
		child_dataset_name: str,
		parent_record_ids: list[int]
) -> int:
	total_records = len(parent_record_ids)
	new_child_dataset_data = ChildDataset(
		name=child_dataset_name,
		approach=approach,
		batch_id=batch_generation_id,
		record_count=total_records
	)

	child_dataset_id = create_child_dataset(new_child_dataset_data).id
	if not child_dataset_id:
		raise ValueError("Could not retrieve Dhild Dataset")

	task_id = create_progress(
		progress_type=ProgressTypeEnum.percentage,
		task_name=f"batch:{batch_generation_id} child:{child_dataset_id}"
	).id


	assert child_dataset_id is not None and task_id is not None

	def background_task() -> None:
		counter = 0

		BATCH_SIZE = config.database.batch_size
		for batch in batch_list(parent_record_ids, BATCH_SIZE):
			data: list[ChildRecord] = []
			for parent_id in batch:
				new = ChildRecord(
					child_id=child_dataset_id,
					batch_id=batch_generation_id,
					parent_record_id=parent_id
				)
				data.append(new)
			
			bulk_create_child_records(data)
			counter += len(batch)

			post_progress(task_id, total_records, counter)
		
		finish_progress(task_id)

	threading.Thread(target=background_task).start()
	return task_id

def generate_processed_datasets(data: ProcessedDatasetCreation) -> list[ProcessedDatasetCreationResponse]:
	response = []
	
	batch_name = f"batch-{datetime.now()}"
	generation_batch_id = create_generation_batch(GenerationBatch(name=data.batch_name or batch_name)).id

	if not generation_batch_id:
		raise ValueError("Could not retrieve Generation Batch")

	total_to_create = sum(dataset.size for dataset in data.datasets)

	parent_records_ids = get_parents_records_ids_by_approach(
		approach=data.approach,
		total_amount=total_to_create
	)

	random.seed(data.seed)
	random.shuffle(parent_records_ids)
	records_split_start = 0

	for child in data.datasets:
		records_split_end = records_split_start + child.size
		parent_ids_to_use = parent_records_ids[records_split_start:records_split_end]

		task_id = background_child_dataset_creation(
			approach=data.approach,
			batch_generation_id=generation_batch_id,
			child_dataset_name=child.name,
			parent_record_ids=parent_ids_to_use
		)

		records_split_start = records_split_end

		response.append(ProcessedDatasetCreationResponse(
			name=child.name,
			task_id=task_id
		))

	return response