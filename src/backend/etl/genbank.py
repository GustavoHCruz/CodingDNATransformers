from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData
from etl.utils import post_progress
from services.progress_tracker_service import finish_progress
from services.raw_file_info_service import save_file


def exin_classifier_gb(genbank_file_path: str, total_records: int | None, new_reading, task_id: int, parent_id: int, seq_max_len=512, flank_max_len=25):
	record_counter = 0
	threshold = 5000

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			if record_counter % threshold == 0:
				post_progress(task_id, new_reading, total_records, record_counter)
			sequence = record.seq

			if (isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData))):
				record_counter += 1
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
					feature_sequence = str(feature_sequence) if strand == 1 else str(feature_sequence.reverse_complement())

					if len(feature_sequence) > seq_max_len:
						continue
					
					before = ""
					start = location.start - flank_max_len
					if start < 0:
						start = 0
					if location.start > 0:
						before = sequence[start:location.start]
						before = str(before) if strand == 1 else str(before.reverse_complement())

					after = ""
					end = location.end + flank_max_len
					if end > len(sequence):
						end = len(sequence)
					if location.end < len(sequence):
						after = sequence[location.end:end]
						after = str(after) if strand == 1 else str(after.reverse_complement())

					label = str(feature.type)

					organism = str(organism)
					gene = str(gene[0] if type(gene) == list else gene)

					yield dict(
						parent_id=parent_id,
						sequence=feature_sequence,
						target=label,
						flank_before=before,
						flank_after=after,
						organism=organism,
						gene=gene
					)

			record_counter += 1

	if new_reading:
		save_file(genbank_file_path, record_counter)
	
	post_progress(task_id, new_reading, total_records, record_counter)
	finish_progress(task_id)

def exin_translator_gb(genbank_file_path: str, total_records: int | None, new_reading, task_id: int, parent_id: int, seq_max_len=512):
	record_counter = 0
	threshold = 5000

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			if record_counter % threshold == 0:
				post_progress(task_id, new_reading, total_records, record_counter)
			sequence = record.seq

			if len(sequence) > seq_max_len or isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)):
				record_counter += 1
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

			yield dict(
				parent_id=parent_id,
				sequence=sequence,
				target=final_sequence,
				organism=organism
			)

			record_counter += 1

	if new_reading:
		save_file(genbank_file_path, record_counter)
	
	post_progress(task_id, new_reading, total_records, record_counter)
	finish_progress(task_id)

def sliding_window_tagger_gb(genbank_file_path: str, total_records: int | None, new_reading, task_id: int, parent_id: int, seq_max_len=512):
	record_counter = 0
	threshold = 5000

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			if record_counter % threshold == 0:
				post_progress(task_id, new_reading, total_records, record_counter)
			sequence = record.seq

			if isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)) or len(sequence) > seq_max_len or len(sequence) < 3:
				record_counter += 1
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
				continue

			rest = len(sequence) - len(splicing_seq)
			if rest:
				splicing_seq += ["U"] * rest

			sequence = str(sequence) if strand == 1 else str(sequence.reverse_complement())

			yield dict(
				parent_id=parent_id,
				sequence=sequence,
				target="".join(splicing_seq),
				organism=organism
			)

			record_counter += 1

	if new_reading:
		save_file(genbank_file_path, record_counter)
	
	post_progress(task_id, new_reading, total_records, record_counter)
	finish_progress(task_id)

def protein_translator_gb(genbank_file_path: str, total_records: int | None, new_reading, task_id: int, parent_id: int, seq_max_len=512):
	record_counter = 0
	threshold = 5000

	with open(genbank_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			if record_counter % threshold == 0:
				post_progress(task_id, new_reading, total_records, record_counter)
			sequence = record.seq

			if isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)) or len(sequence) > seq_max_len or len(sequence) < 3:
				record_counter += 1
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

			yield dict(
				parent_id=parent_id,
				sequence=str(sequence),
				target=str(translation[0]),
				organism=str(organism)
			)

			record_counter += 1

	if new_reading:
		save_file(genbank_file_path, record_counter)
	
	post_progress(task_id, new_reading, total_records, record_counter)
	finish_progress(task_id)