from services.progress_tracker_service import set_progress


def post_progress(task_id: int, new_reading: bool, total_records: int, counter: int) -> None:
	progress = counter
	if not new_reading:
		progress = (counter * 100) / total_records

	set_progress(task_id, progress)