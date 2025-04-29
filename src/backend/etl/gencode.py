from Bio import SeqIO
from services.progress_tracker_service import finish_progress
from services.raw_file_info_service import save_file


def load_fasta(genome_file_path):
	sequences = {}
	for record in SeqIO.parse(genome_file_path, "fasta"):
		sequences[record.id] = str(record.seq)
	return sequences

def parse_gtf(gtf_file_path):
	with open(gtf_file_path, "r", encoding="utf-8") as gtf_file:
		for line in gtf_file:
			line = line.strip()
			if not line or line.startswith("#"):
				continue

			parts = line.split('\t')
			if len(parts) != 9:
				continue

			chrom, source, feature, start, end, _, strand, _, attrs = parts

			attributes = {}
			for attr in attrs.strip().split(';'):
				if attr.strip():
					key_value = attr.strip().split(' ', 1)
					if len(key_value) == 2:
						key, value = key_value
						attributes[key] = value.strip().strip('"')
			
			yield {
				"chrom": chrom,
				"source": source,
				"feature": feature,
				"start": int(start),
				"end": int(end),
				"strand": strand,
				"attributes": attributes
			}

def reverse_complement(sequence, strand):
	complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

	if strand == "-":
		return ''.join(complement[base] for base in reversed(sequence))
	
	return sequence

def exin_classifier_gc(gencode_fasta_file_path: str, gencode_annotations_file_path: str, parent_id: int, seq_max_len=512, flank_max_len=25):
	record_counter = 0
	transcripts = {}

	fasta_sequences = load_fasta(gencode_fasta_file_path)

	for annotation in parse_gtf(gencode_annotations_file_path):
		record_counter += 1

		feature = annotation.get("feature", None)
		attributes = annotation.get("attributes", None)
		
		if feature != "exon":
			continue

		if not attributes:
			continue

		transcript_id = attributes.get("transcript_id", None)
		gene = attributes.get("gene_name", "")
		strand = annotation.get("strand")
		
		if not transcript_id:
			continue

		start, end = int(annotation.get("start")) - 1, int(annotation.get("end"))
		if end-start > seq_max_len:
			continue
		chrom = annotation.get("chrom", None)
		
		if chrom not in fasta_sequences:
			continue
		
		if transcript_id not in transcripts:
			transcripts[transcript_id] = []
		transcripts[transcript_id].append((chrom, start, end, strand, gene))

	for transcript_id, exons in transcripts.items():
		exons.sort(key=lambda x: x[1])

		for i, (chrom, start, end, strand, gene) in enumerate(exons):
			seq = fasta_sequences.get(chrom, "")[start:end]

			flank_before = fasta_sequences.get(chrom, "")[max(0, start-flank_max_len):start]
			flank_after = fasta_sequences.get(chrom, "")[end:end+flank_max_len]

			yield dict(
				parent_id=parent_id,
				sequence=str(reverse_complement(seq, strand)),
				target="exon",
				flank_before=str(reverse_complement(flank_before, strand)),
				flank_after=str(reverse_complement(flank_after, strand)),
				organism=str("Homo sapiens"),
				gene=str(gene),
			)

			if i < len(exons) - 1:
				next_start = exons[i + 1][1]
				if next_start-end > seq_max_len:
					continue
				intron_seq = fasta_sequences.get(chrom, "")[end:next_start]

				if intron_seq:
					flank_before = fasta_sequences.get(chrom, "")[max(0, end - flank_max_len):end]
					flank_after = fasta_sequences.get(chrom, "")[next_start:next_start+flank_max_len]

					yield dict(
						parent_id=parent_id,
						sequence=str(reverse_complement(intron_seq, strand)),
						target="intron",
						flank_before=str(reverse_complement(flank_before, strand)),
						flank_after=str(reverse_complement(flank_after, strand)),
						organism=str("Homo sapiens"),
						gene=str(gene),
					)