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

def splicing_sites_extraction(genbank_file, csv_output_file):
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
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", position=0, leave=True)

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
						before = sequence[location.start-10:location.start]
					after = ""
					if location.end < len(sequence):
						after = sequence[location.end+1:location.end+1]

					if len(feature_sequence) > 512:
						continue
					label = feature.type

					data.append({
						"sequence": str(feature_sequence),
						"label": label,
						"organism": organism,
						"gene": gene,
						"before": str(before),
						"after": str(after)
					})

			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		fieldnames = ["sequence", "label", "organism", "gene", "before", "after"]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		writer.writerows(data)
	
	if new_reading:
		df.loc[len(df)] = [genbank_file, record_counter]
		df.to_csv(cache_file_path, index=False)

splicing_sites_extraction("datasets/exons_and_introns.gb", "datasets/exons_and_introns.csv")