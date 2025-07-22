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

			cds_regions = []
			for feature in record.features:
				if feature.type == "CDS" and feature.qualifiers.get("partial", ["false"])[0] == "false":
					cds_regions.append(feature.location)
					
			for feature in record.features:
				if feature.type not in ["intron", "exon"]:
					continue

				if "pseudo" in feature.qualifiers or feature.qualifiers.get("partial", ["false"][0] == "true"):
					continue

				location = feature.location

				strand = location.strand
				if strand is None:
					strand = feature.qualifiers.get("strand", [None])[0]
				if strand is None:
					continue

				is_inside_cds = any(location.start >= cds.start and location.end <= cds.end for cds in cds_regions)
				if not is_inside_cds:
					continue

				feature_sequence = sequence[location.start:location.end]
				if strand == -1:
					feature_sequence = feature_sequence.reverse_complement()
				feature_sequence = str(feature_sequence)

				if len(feature_sequence) > seq_max_len or len(feature_sequence) == 0:
					continue
					
				before = sequence[max(0, location.start - flank_max_len):location.start]
				after = sequence[location.end:min(len(sequence), location.end + flank_max_len)]

				if strand == -1:
					before = before.reverse_complement()
					after = after.reverse_complement()
				
				before = str(before)
				after = str(after)

				gene = feature.qualifiers.get("gene", "")
				gene = gene[0] if isinstance(gene, list) else gene

				target = str(feature.type)
				organism = str(organism)

				yield dict(
					sequence=feature_sequence,
					target=target,
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
