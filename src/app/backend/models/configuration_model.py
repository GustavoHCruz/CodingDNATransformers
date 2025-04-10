from pydantic import BaseModel


class ConfigModel(BaseModel):
	datasets_raw_dir: str
	datasets_dir: str
	cache_dir: str
	cache_file: str
	genbank_file: str
	gencode_file_annotations: str
	gencode_file_genome: str