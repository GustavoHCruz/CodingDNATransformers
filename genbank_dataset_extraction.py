import csv
import os

import pandas as pd
from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

def splicing_sites_extraction(genbank_file, csv_output_file, seq_max_len=512, flank_len=10):
	total_records = None

	cache_file_path = "cache/genbank_files_len"

	if not os.path.isdir("cache"):
		os.makedirs("cache")

	new_reading = True
	cache_file_status = os.path.isfile(cache_file_path)
	if not cache_file_status:
		df = pd.DataFrame({"gb_name": [], "total_records": []})
	else:
		df = pd.read_csv(cache_file_path)

		filtered_index = df.index[df["gb_name"] == genbank_file]

		if not filtered_index.empty:
			total_records = df.at[filtered_index[0], "total_records"]
			new_reading = False

	data = []
	record_counter = 0
	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", position=0, leave=True, ncols=100)

	with open(genbank_file, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			sequence = record.seq

			if (isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData))):
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			organism = record.annotations.get("organism", "")

			for feature in record.features:
				if feature.type in ["intron", "exon"]:
					location = feature.location
					gene = feature.qualifiers.get("gene", "")
					feature_sequence = sequence[location.start:location.end]
					before = ""
					if location.start > 0:
						before = sequence[location.start-flank_len:location.start]
					after = ""
					if location.end < len(sequence):
						after = sequence[location.end+1:location.end+1+flank_len]

					if len(feature_sequence) > seq_max_len:
						continue
					label = feature.type

					data.append({
						"sequence": str(feature_sequence),
						"label": str(label),
						"organism": str(organism),
						"gene": str(gene),
						"flank_before": str(before),
						"flank_after": str(after)
					})

			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	unique_records = set()
	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		fieldnames = ["sequence", "label", "organism", "gene", "flank_before", "flank_after"]

		progress_bar = tqdm(total=len(data), desc="Writing CSV Progress", position=0, leave=True, ncols=100)

		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		duplicated_counter = 0
		for record in data:
			record_tuple = tuple(record[field] for field in fieldnames)

			if record_tuple not in unique_records:
				unique_records.add(record_tuple)
				writer.writerow(record)
			else:
				duplicated_counter += 1
			
			progress_bar.update(1)
			progress_bar.set_postfix_str(f"{duplicated_counter} duplicated records ignored")

	if new_reading:
		df.loc[len(df)] = [genbank_file, record_counter]
		df.to_csv(cache_file_path, index=False)
