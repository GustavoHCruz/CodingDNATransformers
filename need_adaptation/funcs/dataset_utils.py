import csv
import os

import pandas as pd
from tqdm import tqdm

from funcs.config_reading import read_config_file


def initial_configuration(caller):
	config = read_config_file()
	dirs = ["datasets_raw_dir", "datasets_processed_dir",  "datasets_dir", "cache_dir"]

	for directory in dirs:
		if not os.path.isdir(config[directory]):
			os.makedirs(config[directory])

	cache_file_path = f"{config["cache_dir"]}/{config["cache_file"]}"
	
	if caller == "genbank":
		genbank_file_path = f"{config["datasets_raw_dir"]}/{config["genbank_file"]}"

		return config, genbank_file_path, cache_file_path

	elif caller == "gencode":
		gencode_genome_file_path = f"{config["datasets_raw_dir"]}/{config["gencode_file_genome"]}"
		gencode_annotations_file_path = f"{config["datasets_raw_dir"]}/{config["gencode_file_annotations"]}"

		return config, gencode_genome_file_path, gencode_annotations_file_path, cache_file_path
	
	raise ValueError("Unable to determine caller")

def cache_initial_config(file_path, cache_file_path):
	total_records = None
	cache_file_status = os.path.isfile(cache_file_path)
	if not cache_file_status:
		df = pd.DataFrame({"file_name": [], "total_records": []})
	else:
		df = pd.read_csv(cache_file_path)

		filtered_index = df.index[df["file_name"] == file_path]

		if not filtered_index.empty:
			total_records = df.at[filtered_index[0], "total_records"]
	
	return total_records, total_records == None

def cache_save_config(record_counter, file_path, cache_file_path):
	df = pd.read_csv(cache_file_path)

	df.loc[len(df)] = [file_path, record_counter]
	df.to_csv(cache_file_path, index=False)
	
def write_csv(csv_output_file, fieldnames, data):
	unique_records = set()
	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		progress_bar = tqdm(total=len(data), desc="Writing CSV Progress", leave=True)

		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		duplicated_counter = 0
		counter = 0
		for record in data:
			record_tuple = tuple(record[field] for field in fieldnames)
			record_hash = hash(record_tuple)
			
			if record_hash not in unique_records:
				unique_records.add(record_hash)
				writer.writerow(record)
			else:
				duplicated_counter += 1
			counter += 1
			
			if counter % 1000 == 0:
				progress_bar.update(1000)
				progress_bar.set_postfix_str(f"{duplicated_counter} duplicated")

		progress_bar.update(counter % 1000)
	
	progress_bar.close()