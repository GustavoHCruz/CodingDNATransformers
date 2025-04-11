import hashlib


def generate_sequence_hash(*args: str) -> str:
  concat_str = "|".join(args)
  return hashlib.sha256(concat_str.encode()).hexdigest()