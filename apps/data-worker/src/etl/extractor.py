
from Bio import SeqIO
from Bio.Seq import _PartiallyDefinedSequenceData, _UndefinedSequenceData
from Bio.SeqFeature import CompoundLocation

parsed_records = []

gb_file = "seu_arquivo.gb"

def parse_feature_location(loc) -> tuple[int, int]:
	return int(loc.start), int(loc.end)

# ==========================
records_ignored = 0
cds_ignored = 0
exin_ignored = 0
# ==========================
# ==========================
flanks_max_length = 25
# ==========================


for record in SeqIO.parse(gb_file, "genbank"):
	if (isinstance(record.seq._data, (_UndefinedSequenceData, _PartiallyDefinedSequenceData))):
		continue

	sequence_dna = record.seq
	accession = str(record.id)
	organism = str(record.annotations.get("organism", ""))

	cds_regions = []
	exin = []

	for feature in record.features:
		if isinstance(feature.location, CompoundLocation) and any(part.ref for part in feature.location.parts):
			cds_ignored += 1
			continue

		translation = feature.qualifiers.get("translation", None)
		if not translation:
			cds_ignored += 1
			continue

		cds_seq = feature.extract(record.seq)

		if len(cds_seq < 3):
			cds_ignored += 1
			continue

		if isinstance(cds_seq._data, _UndefinedSequenceData):
			cds_ignored += 1
			continue

		location = feature.location
		start = int(location.start)
		end = int(location.end)

		gene = feature.qualifiers.get("gene", "")

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
				exin_ignored += 1
				continue

			strand = None
			if hasattr(location, "strand"):
				strand = location.strand
			if strand is None:
				exin_ignored += 1
				continue
			
			feature_sequence = sequence_dna[location.start:location.end]
			feature_sequence = str(feature_sequence) if strand == 1 else str(feature_sequence.reverse_complement())

			if len(feature_sequence) < 3:
				exin_ignored += 1
				continue

			before = sequence_dna[max(0, location.start - flanks_max_length):location.start]
			after = sequence_dna[location.end:min(len(sequence_dna), location.end + flanks_max_length)]

			if strand == 1:
				before = str(before)
				after = str(after)
			else:
				before = before.reverse_complement()
				after = after.reverse_complement()
			
			label = str(feature.type)
			gene = str(gene[0] if type(gene) == list else gene)

			is_inside_cds = any(location.start >= cds["start"] and location.end <= cds["end"
			] for cds in cds_regions)
			if not is_inside_cds:
				continue

			exin.append({
				"sequence": feature_sequence,
				"type": label,
				"start": int(location.start),
				"end": int(location.end),
				"gene": gene,
				"strand": strand,
				"before": before,
				"after": after
			})
	
	parsed_records.append({
		"sequence": str(sequence_dna),
		"accession": accession,
		"organism": organism,
		"cds_regions": cds_regions,
		"exin": exin
	})

print(f"Total de registros processados: {len(parsed_records)}")
print(f"Total ignored: {records_ignored}")
print(f"Exin ignored: {exin_ignored}")
print(f"cds ignored: {cds_ignored}")