from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData
from tqdm import tqdm

from funcs.config_reading import read_datasets_configs
from funcs.dataset_utils import (cache_initial_config, cache_save_config,
                                 initial_configuration, write_csv)


def exin_classifier(seq_max_len=512, flank_max_len=25):
	caller = "genbank"
	config, genbank_file_path, cache_file_path = initial_configuration(caller)
	approach = "ExInSeqs"
	csv_output_file = f"{config["datasets_processed_dir"]}/{approach}_{caller}.csv"

	total_records, new_reading = cache_initial_config(genbank_file_path, cache_file_path)

	data = []
	record_counter = 0
	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, caller):
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
					start = location.start - flank_max_len
					if start < 0:
						start = 0
					if location.start > 0:
						before = sequence[start:location.start]

					after = ""
					end = location.end + flank_max_len
					if end > len(sequence):
						end = len(sequence)
					if location.end < len(sequence):
						after = sequence[location.end:end]

					label = feature.type

					data.append({
						"sequence": str(feature_sequence) if strand == 1 else str(feature_sequence.reverse_complement()),
						"label": str(label),
						"organism": str(organism),
						"gene": str(gene[0] if type(gene) == list else gene),
						"flank_before": str(before) if strand == 1 else str(before.reverse_complement()),
						"flank_after": str(after) if strand == 1 else str(after.reverse_complement()),
					})

			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)
		
	progress_bar.close()

	fieldnames = ["sequence", "label", "organism", "gene", "flank_before", "flank_after"]
	write_csv(csv_output_file=csv_output_file, fieldnames=fieldnames, data=data)

	if new_reading:
		cache_save_config(record_counter, genbank_file_path, cache_file_path)

def sequence_rebuild_extraction():
	caller = "genbank"
	config, genbank_file_path, cache_file_path = initial_configuration(caller)
	approach = "RebuilSeqs"
	csv_output_file = f"{config["datasets_processed_dir"]}/{approach}_genbank.csv"
	datasets_configs = read_datasets_configs(approach)

	seq_max_len = datasets_configs["default"]["sequence_length"]

	total_records, new_reading = cache_initial_config(genbank_file_path, cache_file_path)

	data = []
	record_counter = 0
	accepted_records_counter = 0

	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, caller):
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

	progress_bar.close()

	fieldnames = ["sequence", "builded", "organism"]
	write_csv(csv_output_file=csv_output_file, fieldnames=fieldnames, data=data)

	if new_reading:
		cache_save_config(record_counter, genbank_file_path, cache_file_path)

def sliding_window_extraction():
	caller = "genbank"
	config, genbank_file_path, cache_file_path = initial_configuration(caller)
	approach = "SWExInSeqs"
	csv_output_file = f"{config["datasets_processed_dir"]}/{approach}_genbank.csv"
	datasets_configs = read_datasets_configs(approach)

	seq_max_len = datasets_configs["default"]["sequence_length"]

	total_records, new_reading = cache_initial_config(genbank_file_path, cache_file_path)

	data = []
	record_counter = 0
	accepted_records_counter = 0

	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, caller):
			sequence = record.seq

			if isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)) or len(sequence) > seq_max_len or len(sequence) < 3:
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			organism = record.annotations.get("organism", "")

			splicing_seq = ["U"] * len(sequence)
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
						
					char = "I" if feature.type == "intron" else "E"
					splicing_seq[start:end] = [char] * (end - start)

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
				"labeled_sequence": "".join(splicing_seq)
			})

			accepted_records_counter += 1
			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	print(f"Accepted Records: {accepted_records_counter}")

	progress_bar.close()

	fieldnames = ["sequence", "organism", "labeled_sequence"]
	write_csv(csv_output_file=csv_output_file, fieldnames=fieldnames, data=data)

	if new_reading:
		cache_save_config(record_counter, genbank_file_path, cache_file_path)

def protein_extraction():
	caller = "genbank"
	config, genbank_file_path, cache_file_path = initial_configuration(caller)
	approach = "ProteinSeqs"
	csv_output_file = f"{config["datasets_processed_dir"]}/{approach}_genbank.csv"
	datasets_configs = read_datasets_configs(approach)

	seq_max_len = datasets_configs["default"]["sequence_length"]

	total_records, new_reading = cache_initial_config(genbank_file_path, cache_file_path)

	data = []
	record_counter = 0

	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan Progress", leave=True)

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, caller):
			sequence = record.seq

			if isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)) or len(sequence) > seq_max_len or len(sequence) < 3:
				record_counter += 1
				if not new_reading:
					progress_bar.update(1)
				continue

			organism = record.annotations.get("organism", "")

			allow = False
			for feature in record.features:
				if feature.type == "CDS":
					allow = True
					translation = feature.qualifiers.get("translation", None)
					break
			
			if not allow or not translation:
				record_counter += 1
				continue

			data.append({
				"sequence": str(sequence),
				"organism": str(organism),
				"translation": str(translation[0])
			})

			record_counter += 1

			if new_reading:
				progress_bar.set_description_str(f"Records Scanned: {record_counter}")
			else:
				progress_bar.update(1)

	progress_bar.close()

	fieldnames = ["sequence", "organism", "translation"]
	write_csv(csv_output_file=csv_output_file, fieldnames=fieldnames, data=data)

	if new_reading:
		cache_save_config(record_counter, genbank_file_path, cache_file_path)

splicing_sites_extraction()