from Bio import SeqIO
from Bio.Seq import Seq


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

def exin_classifier_gc(fasta_file_path: str, annotations_file_path: str, seq_max_len=512, flank_max_len=25):
	record_counter = 0
	transcripts = {}

	fasta_sequences = load_fasta(fasta_file_path)

	for annotation in parse_gtf(annotations_file_path):
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

		start = annotation.get("start")
		end = annotation.get("end")
		if type(start) == str:
			start = int(start) - 1
		if type(end) == str:
			end = int(end)

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
						sequence=str(reverse_complement(intron_seq, strand)),
						target="intron",
						flank_before=str(reverse_complement(flank_before, strand)),
						flank_after=str(reverse_complement(flank_after, strand)),
						organism=str("Homo sapiens"),
						gene=str(gene),
					)

def exin_translator_gc(fasta_file_path: str, annotations_file_path: str, seq_max_len=512):
	record_counter = 0
	transcripts = {}

	fasta_sequences = load_fasta(fasta_file_path)

	for annotation in parse_gtf(annotations_file_path):
		record_counter += 1

		feature = annotation.get("feature", None)
		attributes = annotation.get("attributes", None)
		
		if feature != "exon":
			continue

		if not attributes:
			continue

		transcript_id = attributes.get("transcript_id", None)
		strand = annotation.get("strand")
		
		if not transcript_id:
			continue

		start = annotation.get("start")
		end = annotation.get("end")
		if type(start) == str:
			start = int(start) - 1
		if type(end) == str:
			end = int(end)

		if end-start > seq_max_len:
			continue
		chrom = annotation.get("chrom", None)
		
		if chrom not in fasta_sequences:
			continue
		
		if transcript_id not in transcripts:
			transcripts[transcript_id] = []
		transcripts[transcript_id].append((chrom, start, end, strand))

	for transcript_id, info in transcripts.items():
		info.sort(key=lambda x: x[1])

		seqs = []
		for i, (chrom, start, end, strand) in enumerate(info):
			seq = fasta_sequences.get(chrom, "")[start:end]

			seqs.append(dict(
				sequence=str(reverse_complement(seq, strand)),
				target="exon",
			))

			if i < len(info) - 1:
				next_start = info[i + 1][1]
				if next_start-end > seq_max_len:
					continue
				intron_seq = fasta_sequences.get(chrom, "")[end:next_start]

				if intron_seq:
					seqs.append(dict(
						sequence=str(reverse_complement(intron_seq, strand)),
						target="intron",
					))
		
		final_seq = "".join(seq["sequence"] for seq in seqs)
		final_target = "".join(f"({seq['target']}){seq['sequence']}({seq['target']})" for seq in seqs)

		if len(final_seq) < seq_max_len:		
			yield dict(
				sequence=final_seq,
				target=final_target,
				organism="Homo sapiens"
			)

def sliding_window_tagger_gc(fasta_file_path: str, annotations_file_path: str, seq_max_len=512):
	record_counter = 0
	transcripts = {}

	fasta_sequences = load_fasta(fasta_file_path)

	for annotation in parse_gtf(annotations_file_path):
		record_counter += 1

		feature = annotation.get("feature", None)
		attributes = annotation.get("attributes", None)
		
		if feature != "exon" or not attributes:
			continue

		transcript_id = attributes.get("transcript_id", None)
		strand = annotation.get("strand")
		
		if not transcript_id:
			continue

		start = annotation.get("start")
		end = annotation.get("end")
		if type(start) == str:
			start = int(start) - 1
		if type(end) == str:
			end = int(end)

		if end - start > seq_max_len:
			continue

		chrom = annotation.get("chrom", None)
		if chrom not in fasta_sequences:
			continue
		
		if transcript_id not in transcripts:
			transcripts[transcript_id] = []
		transcripts[transcript_id].append((chrom, start, end, strand))

	for transcript_id, info in transcripts.items():
		info.sort(key=lambda x: x[1], reverse=(info[0][3] == "-"))

		sequence_parts = []
		target_parts = []

		for i, (chrom, start, end, strand) in enumerate(info):
			exon_seq = fasta_sequences[chrom][start:end]
			exon_seq = reverse_complement(exon_seq, strand)
			sequence_parts.append(exon_seq)
			target_parts.append("E" * len(exon_seq))

			if i < len(info) - 1:
				next_start = info[i + 1][1]
				intron_len = next_start - end
				if intron_len > seq_max_len or intron_len < 1:
					continue
				intron_seq = fasta_sequences[chrom][end:next_start]
				intron_seq = reverse_complement(intron_seq, strand)
				sequence_parts.append(intron_seq)
				target_parts.append("I" * len(intron_seq))

		final_seq = "".join(sequence_parts)
		final_target = "".join(target_parts)

		if len(final_seq) < seq_max_len:
			yield dict(
				sequence=final_seq,
				target=final_target,
				organism="Homo sapiens"
			)

def protein_translator_gc(fasta_file_path: str, annotations_file_path: str, seq_max_len=512):
	record_counter = 0
	transcripts = {}

	fasta_sequences = load_fasta(fasta_file_path)

	for annotation in parse_gtf(annotations_file_path):
		record_counter += 1

		feature = annotation.get("feature", None)
		attributes = annotation.get("attributes", None)
		
		if feature != "exon" or not attributes:
			continue

		transcript_id = attributes.get("transcript_id", None)
		strand = annotation.get("strand")
		
		if not transcript_id:
			continue

		start = annotation.get("start")
		end = annotation.get("end")
		if type(start) == str:
			start = int(start) - 1
		if type(end) == str:
			end = int(end)
		
		if end - start > seq_max_len:
			continue

		chrom = annotation.get("chrom", None)
		if chrom not in fasta_sequences:
			continue
		
		if transcript_id not in transcripts:
			transcripts[transcript_id] = {"exons": [], "strand": strand}
		transcripts[transcript_id]["exons"].append((chrom, start, end))

	for transcript_id, data in transcripts.items():
		exons = data["exons"]
		strand = data["strand"]
		exons.sort(key=lambda x: x[1], reverse=(strand == "-"))

		sequence = ""
		for chrom, start, end in exons:
			seq = fasta_sequences[chrom][start:end]
			seq = reverse_complement(seq, strand)
			sequence += seq

		cropped_sequence = sequence[:len(sequence) - (len(sequence) % 3)]

		if len(cropped_sequence) < 3:
			continue

		protein = str(Seq(cropped_sequence).translate(to_stop=False))

		if len(protein) < seq_max_len:
			yield dict(
				sequence=cropped_sequence,
				target=protein,
				organism="Homo sapiens"
			)