from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData
from Bio.SeqFeature import CompoundLocation

parsed_records = []

def parse_feature_location(loc) -> tuple[int, int]:
	return int(loc.start), int(loc.end)

def extract_data(
	annotations_file_path: str,
	flanks_max_length = 25
):
	for record in SeqIO.parse(annotations_file_path, "genbank"):
		if (isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData))):
			continue

		sequence_dna = record.seq
		accession = str(record.id)
		organism = str(record.annotations.get("organism", ""))

		cds_regions = []
		exin = []

		for feature in record.features:
			if feature.type == "CDS":
				if isinstance(feature.location, CompoundLocation) and any(part.ref for part in feature.location.parts):
					continue

				translation = feature.qualifiers.get("translation", None)
				if not translation:
					continue

				cds_seq = feature.extract(record.seq)

				if len(str(cds_seq)) < 3:
					continue

				if isinstance(cds_seq._data, _UndefinedSequenceData):
					continue

				location = feature.location
				start = int(location.start)
				end = int(location.end)

				if not start or not end:
					continue

				gene = feature.qualifiers.get("gene", "")
				gene = gene[0] if isinstance(gene, list) else gene

				cds_regions.append({
					"sequence": str(translation[0]),
					"type": "CDS",
					"start": start,
					"end": end,
					"gene": gene
				})

		for feature in record.features:
			if feature.type in ["intron", "exon"]:
				location = feature.location
				gene = feature.qualifiers.get("gene", "")

				if "pseudo" in feature.qualifiers or feature.qualifiers.get("partial", ["false"][0] == "true"):
					continue

				strand = None
				if hasattr(location, "strand"):
					strand = location.strand
				if strand is None:
					continue

				start = location.start
				end = location.end
				if not start or not end:
					continue
				
				feature_sequence = sequence_dna[start:end]
				feature_sequence = str(feature_sequence) if strand == 1 else str(feature_sequence.reverse_complement())

				if len(feature_sequence) < 3:
					continue

				before = sequence_dna[max(0, start - flanks_max_length):start]
				after = sequence_dna[end:min(len(sequence_dna), end + flanks_max_length)]

				if strand == 1:
					before = str(before)
					after = str(after)
				else:
					before = str(before.reverse_complement())
					after = str(after.reverse_complement())
				
				label = str(feature.type)
				gene = str(gene[0] if type(gene) == list else gene)

				is_inside_cds = any(start >= cds["start"] and end <= cds["end"
				] for cds in cds_regions)
				if not is_inside_cds:
					continue

				exin.append({
					"sequence": feature_sequence,
					"type": label.upper(),
					"start": int(start),
					"end": int(end),
					"gene": gene,
					"strand": strand,
					"before": before,
					"after": after
				})
		
		yield dict({
			"sequence": str(sequence_dna),
			"accession": accession,
			"organism": organism,
			"cds": cds_regions,
			"exin": exin
		})
