from repositories.progress_tracker_repo import set_progress


def post_progress(task_id: int, total_records: int | None, counter: int) -> None:
	progress = counter
	if total_records:
		progress = (counter * 100) / total_records

	set_progress(task_id, progress)