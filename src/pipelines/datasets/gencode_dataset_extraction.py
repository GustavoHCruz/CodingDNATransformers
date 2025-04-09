from Bio import SeqIO
from tqdm import tqdm

from funcs.config_reading import read_datasets_configs
from funcs.dataset_utils import (cache_initial_config, cache_save_config,
                                 initial_configuration, write_csv)


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

def splicing_sites_extraction():
	caller = "gencode"
	config, gencode_genome_file_path, gencode_annotations_file_path, cache_file_path = initial_configuration(caller)
	approach = "ExInSeqs"
	csv_output_file = f"{config["datasets_dir"]}/{approach}_{caller}.csv"
	datasets_configs = read_datasets_configs(approach)

	seq_max_len = datasets_configs["default"]["sequence_length"]
	flank_size = datasets_configs["small"]["flanks"]
	flank_extended_size = datasets_configs["default"]["flanks"]

	total_records, new_reading = cache_initial_config(gencode_genome_file_path, cache_file_path)

	data = []
	transcripts = {}
	record_counter = 0
	transcripts_counter = 0
	ignored_ones = 0

	if new_reading:
		progress_bar = tqdm(bar_format="{desc}")
	else:
		progress_bar = tqdm(total=total_records, desc="File Scan progress", leave=True)

	fasta_sequences = load_fasta(gencode_genome_file_path)

	for annotation in parse_gtf(gencode_annotations_file_path):
		record_counter += 1

		if not new_reading:
			progress_bar.update(1)
		else:
			progress_bar.set_description_str(f"Records Scanned: {record_counter}")

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
			ignored_ones += 1
			continue
		chrom = annotation.get("chrom", None)
		
		if chrom not in fasta_sequences:
			continue
		
		if transcript_id not in transcripts:
			transcripts[transcript_id] = []
			transcripts_counter += 1
		transcripts[transcript_id].append((chrom, start, end, strand, gene))
	
	progress_bar.close()

	progress_bar = tqdm(total=transcripts_counter, desc="Processing Data", leave=True)

	for transcript_id, exons in transcripts.items():
		exons.sort(key=lambda x: x[1])

		progress_bar.update(1)

		for i, (chrom, start, end, strand, gene) in enumerate(exons):
			seq = fasta_sequences.get(chrom, "")[start:end]

			flank_before = fasta_sequences.get(chrom, "")[max(0, start-flank_size):start]
			flank_before_extended = fasta_sequences.get(chrom, "")[max(0, start-flank_extended_size):start]
			flank_after = fasta_sequences.get(chrom, "")[end:end+flank_size]
			flank_after_extended = fasta_sequences.get(chrom, "")[end:end+flank_extended_size]

			data.append({
				"sequence": str(reverse_complement(seq, strand)),
				"label": "exon",
				"organism": str("Homo sapiens"),
				"gene": str(gene),
				"flank_before": str(reverse_complement(flank_before, strand)),
				"flank_before_extended": str(reverse_complement(flank_before_extended, strand)),
				"flank_after": str(reverse_complement(flank_after, strand)),
				"flank_after_extended": str(reverse_complement(flank_after_extended, strand))
			})

			if i < len(exons) - 1:
				next_start = exons[i + 1][1]
				if next_start-end > seq_max_len:
					ignored_ones += 1
					continue
				intron_seq = fasta_sequences.get(chrom, "")[end:next_start]

				if intron_seq:
					flank_before = fasta_sequences.get(chrom, "")[max(0, end - flank_size):end]
					flank_before_extended = fasta_sequences.get(chrom, "")[max(0, end - flank_extended_size):end]
					flank_after = fasta_sequences.get(chrom, "")[next_start:next_start+flank_size]
					flank_after_extended = fasta_sequences.get(chrom, "")[next_start:next_start+flank_extended_size]

					data.append({
						"sequence": str(reverse_complement(intron_seq, strand)),
						"label": "intron",
						"organism": str("Homo sapiens"),
						"gene": str(gene),
						"flank_before": str(reverse_complement(flank_before, strand)),
						"flank_before_extended": str(reverse_complement(flank_before_extended, strand)),
						"flank_after": str(reverse_complement(flank_after, strand)),
						"flank_after_extended": str(reverse_complement(flank_after_extended, strand))
					})

	progress_bar.close()

	print(f"A total of {ignored_ones} sequences has been ignored due to the sequence length limit")

	fieldnames = ["sequence", "label", "organism", "gene", "flank_before", "flank_before_extended", "flank_after", "flank_after_extended"]
	write_csv(csv_output_file=csv_output_file, fieldnames=fieldnames, data=data)

	if new_reading:
		cache_save_config(record_counter, gencode_genome_file_path, cache_file_path)