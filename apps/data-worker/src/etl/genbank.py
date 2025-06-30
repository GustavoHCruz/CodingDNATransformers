from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData
from Bio.SeqFeature import CompoundLocation


def exin_classifier_gb(annotations_file_path: str, seq_max_len=512, flank_max_len=25):
	with open(annotations_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			sequence = record.seq

			if (isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData))):
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

					if len(feature_sequence) > seq_max_len or len(feature_sequence) == 0:
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
						sequence=feature_sequence,
						target=label,
						flank_before=before,
						flank_after=after,
						organism=organism,
						gene=gene
					)

def exin_translator_gb(annotations_file_path: str, seq_max_len=512):
	with open(annotations_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			sequence = record.seq

			if len(sequence) > seq_max_len or isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)):
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
				sequence=str(sequence),
				target=final_sequence,
				organism=str(organism),
			)

def sliding_window_tagger_gb(annotations_file_path: str, seq_max_len=512):
	with open(annotations_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			sequence = record.seq

			if isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData)) or len(sequence) > seq_max_len or len(sequence) < 3:
				continue

			organism = record.annotations.get("organism", "")

			splicing_seq = ["U"] * len(sequence)
			strand = None
			for feature in record.features:
				if feature.type in ["intron", "exon"]:
					location = feature.location
			
					strand = None
					if hasattr(location, "strand"):
						strand = location.strand
					if strand is None:
						break

					start = location.start
					end = location.end
					
					if len(splicing_seq) == 0 and start > 0:
						splicing_seq = ["U"] * start
						
					char = "I" if feature.type == "intron" else "E"
					splicing_seq[start:end] = [char] * (end - start)

			if not strand:
				continue

			rest = len(sequence) - len(splicing_seq)
			if rest:
				splicing_seq += ["U"] * rest

			sequence = str(sequence) if strand == 1 else str(sequence.reverse_complement())

			yield dict(
				sequence=sequence,
				target="".join(splicing_seq),
				organism=organism
			)

def protein_translator_gb(annotations_file_path: str, seq_max_len=512):
	with open(annotations_file_path, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			organism = record.annotations.get("organism", "")

			for feature in record.features:
				if feature.type != "CDS":
					continue

				if isinstance(feature.location, CompoundLocation) and any(part.ref for part in feature.location.parts):
					continue

				translation = feature.qualifiers.get("translation")
				if not translation:
					continue

				cds_seq = feature.extract(record.seq)

				if len(cds_seq) < 3 or len(cds_seq) > seq_max_len:
					continue

				if isinstance(cds_seq._data, _UndefinedSequenceData):
					continue

				yield {
					"sequence": str(cds_seq),
					"target": str(translation[0]),
					"organism": organism
				}
