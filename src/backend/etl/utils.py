import hashlib

from services.progress_tracker_service import set_progress


def generate_hash(*args: str) -> str:
  concat_str = "|".join(args)
  return hashlib.sha256(concat_str.encode()).hexdigest()


def post_progress(task_id: int, new_reading: bool, total_records: int, counter: int) -> None:
	progress = counter
	if not new_reading:
		progress = total_records / counter

	set_progress(task_id, progress)