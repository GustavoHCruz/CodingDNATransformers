import os

from funcs.config_reading import read_config_file


def update_dirs():
	config = read_config_file()
	dirs = ["datasets_raw_dir", "datasets_processed_dir",  "datasets_dir", "cache_dir"]

	for directory in dirs:
		if not os.path.isdir(config[directory]):
			os.makedirs(config[directory])