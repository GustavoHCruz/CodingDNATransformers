import csv
import os

import pandas as pd
from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData

from funcs.config_reading import read_datasets_configs

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

def splicing_sites_extraction(genbank_file, csv_output_file, seq_max_len=512):
	datasets_configs = read_datasets_configs()

	flank_len = datasets_configs["version"]["small"]["flanks"]
	flank_extended_len = datasets_configs["version"]["normal"]["flanks"]
	total_records = None

	cache_file_path = "cache/genbank_files_len.csv"

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
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

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

					strand = None
					if hasattr(location, "strand"):
						strand = location.strand
					if strand is None:
						continue

					feature_sequence = sequence[location.start:location.end]

					if len(feature_sequence) > seq_max_len:
						continue
					
					before = ""
					before_extended = ""
					start = location.start - flank_len
					start_extended = location.start - flank_extended_len
					if start < 0:
						start = 0
					if start_extended < 0:
						start_extended = 0
					if location.start > 0:
						before = sequence[start:location.start]
						before_extended = sequence[start_extended:location.start]

					after = ""
					after_extended = ""
					end = location.end + flank_len
					end_extended = location.end + flank_extended_len
					if end > len(sequence):
						end = len(sequence)
					if end_extended > len(sequence):
						end_extended = len(sequence)
					if location.end < len(sequence):
						after = sequence[location.end:end]
						after_extended = sequence[location.end:end_extended]

					label = feature.type

					data.append({
						"sequence": str(feature_sequence) if strand == 1 else str(feature_sequence.reverse_complement()),
						"label": str(label),
						"organism": str(organism),
						"gene": str(gene),
						"flank_before": str(before),
						"flank_before_extended": str(before_extended),
						"flank_after": str(after),
						"flank_after_extended": str(after_extended),
					})

			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	unique_records = set()
	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		fieldnames = ["sequence", "label", "organism", "gene", "flank_before", "flank_before_extended", "flank_after", "flank_after_extended"]

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

	if new_reading:
		df.loc[len(df)] = [genbank_file, record_counter]
		df.to_csv(cache_file_path, index=False)

def sequence_rebuild_extraction(genbank_file, csv_output_file, seq_max_len=512):
	total_records = None

	cache_file_path = "cache/genbank_files_len.csv"

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
	accepted_records_counter = 0

	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

	with open(genbank_file, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			sequence = record.seq

			if len(sequence) > seq_max_len or isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)):
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			organism = record.annotations.get("organism", "")

			sites = []
			strand_invalid = False
			for feature in record.features:
				if feature.type in ["intron", "exon"]:
					location = feature.location
			
					strand = None
					if hasattr(location, "strand"):
						strand = location.strand
					if strand is None:
						strand_invalid = True
						break

					site_seq = sequence[location.start:location.end]
					if strand == -1:
						site_seq = site_seq.reverse_complement()

					sites.append({
						"sequence": site_seq,
						"start": location.start,
						"end": location.end,
						"type": "intron" if feature.type == "intron" else "exon",
					})
			
			if strand_invalid:
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			final_sequence = []
			last_index = 0
			for site in sorted(sites, key=lambda x: x["start"]):
				final_sequence.append(str(sequence[last_index:site["start"]]))
				final_sequence.append(f"({site["type"]})")
				final_sequence.append(str(sequence[site["start"]:site["end"]]))
				final_sequence.append(f"({site["type"]})")
				last_index = site["end"]
			
			final_sequence.append(str(sequence[last_index:]))
			final_sequence = "".join(final_sequence)

			data.append({
				"sequence": sequence,
				"builded": final_sequence,
				"organism": organism
			})

			accepted_records_counter += 1
			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	print(f"Accepted Records: {accepted_records_counter}")

	unique_records = set()
	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		fieldnames = ["sequence", "builded", "organism"]

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

	if new_reading:
		df.loc[len(df)] = [genbank_file, record_counter]
		df.to_csv(cache_file_path, index=False)

def sliding_window_extraction(genbank_file, csv_output_file):
	total_records = None

	cache_file_path = "cache/genbank_files_len.csv"

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
	accepted_records_counter = 0

	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

	with open(genbank_file, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			sequence = record.seq

			if isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)):
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			organism = record.annotations.get("organism", "")

			splicing_seq = []
			strand_invalid = False
			for feature in record.features:
				if feature.type in ["intron", "exon"]:
					location = feature.location
			
					strand = None
					if hasattr(location, "strand"):
						strand = location.strand
					if strand is None:
						strand_invalid = True
						break

					start = location.start
					end = location.end
					
					if len(splicing_seq) == 0 and start > 0:
						splicing_seq = ["U"] * start
						
					if feature.type == "intron":
						splicing_seq += ["I"] * (end-start)
					else:
						splicing_seq += ["E"] * (end-start)

			if strand_invalid:
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			rest = len(sequence) - len(splicing_seq)
			if rest:
				splicing_seq += ["U"] * rest

			sequence = str(sequence) if strand == 1 else str(sequence.reverse_complement())

			data.append({
				"sequence": sequence,
				"organism": organism,
				"labeled_sequence": splicing_seq
			})

			accepted_records_counter += 1
			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	print(f"Accepted Records: {accepted_records_counter}")

	unique_records = set()
	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		fieldnames = ["sequence", "organism", "labeled_sequence"]

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

	if new_reading:
		df.loc[len(df)] = [genbank_file, record_counter]
		df.to_csv(cache_file_path, index=False)