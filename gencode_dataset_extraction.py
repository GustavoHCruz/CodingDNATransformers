import csv
import os
import re

import pandas as pd
from Bio import SeqIO

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

def load_fasta(fasta_file):
	sequences = {}
	for record in SeqIO.parse(fasta_file, "fasta"):
		sequences[record.id] = str(record.seq)
	return sequences

def parse_gtf_attributes(attribute_string):
	attributes = {}
	matches = re.findall(r'(\S+) "([^"]+)"', attribute_string)
	for key, value in matches:
		attributes[key] = value
	return attributes

def reverse_complement(sequence, strand):
	complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

	if strand == 1:
		return ''.join(complement[base] for base in reversed(sequence))
	
	return sequence

def process_gtf(gtf_file, fasta_file, csv_output_file):
	fasta_sequences = load_fasta(fasta_file)

	datasets_configs = read_datasets_configs("ExInSeqs", "gencode")

	seq_max_len = datasets_configs["version"]["default"]["sequence_length"]
	flank_size = datasets_configs["version"]["small"]["flanks"]
	flank_extended_size = datasets_configs["version"]["default"]["flanks"]
	total_records = None

	cache_file_path = "cache/gencode_files_len.csv"

	if not os.path.isdir("cache"):
		os.makedirs("cache")

	new_reading = True
	cache_file_status = os.path.isfile(cache_file_path)
	if not cache_file_status:
		df = pd.DataFrame({"gc_name": [], "total_records": []})
	else:
		df = df.read_csv(cache_file_path)

		filtered_index = df.index[df["gb_name"] == gtf_file]

		if not filtered_index.empty:
			total_records = df.at[filtered_index[0], "total_records"]
			new_reading = False

		data = []
		transcripts = {}
		record_counter = 0
		if new_reading:
			progress_bar = tqdm(bar_format="{desc}")
		else:
			progress_bar = tqdm(total=total_records, desc="File Scan progress", leave=True)
		
		with open(gtf_file, 'r') as gtf:
			for line in gtf:
				if line.startswith("#"):
					continue
				fields = line.strip().split('\t')
				if len(fields) < 9:
					continue
				
				chrom, _, feature, start, end, _, strand, _, attributes = fields

				if feature not in ["exon"]:
					continue

				attributes = parse_gtf_attributes(attributes)
				gene = attributes.get("gene_name", "unknown")
				transcript_id = attributes.get("transcript_id", "unknown")
				
				start, end = int(start) - 1, int(end)

				seq = fasta_sequences.get(chrom, "")[start:end]

				before = fasta_sequences.get(chrom, "")[max(0, start-flank_size):start]
				before_extended = fasta_sequences.get(chrom, "")[max(0, start-flank_extended_size):start]

				after = fasta_sequences.get(chrom, "")[end:end+flank_size]
				after_extended = fasta_sequences.get(chrom, "")[end:end+flank_extended_size]
				
				if transcript_id not in transcripts:
					transcripts[transcript_id] = []
				transcripts[transcript_id].append((start, end))

				data.append({
					"sequence": str(reverse_complement(seq, strand)),
					"label": "exon",
					"organism": str("Homo sapiens"),
					"gene": str(gene),
					"flank_before": str(before),
					"flank_before_extended": str(before_extended),
					"flank_after": str(after),
					"flank_after_extended": str(after_extended)
				})

				record_counter += 1

				if new_reading:
					progress_bar.set_description_str(f"Records Scanned: {record_counter}")
				else:
					progress_bar.update(1)
			
	
				
	
		# Ordenar os éxons por posição
		for transcript_id, exons in transcripts.items():
				exons.sort(key=lambda x: x[1])
				
				for i, (chrom, start, end, strand) in enumerate(exons):
						seq = fasta_sequences.get(chrom, "")[start:end]
						if not seq:
								continue
						
						# Definir flancos
						flank_left = fasta_sequences.get(chrom, "")[max(0, start-flank_size):start]
						flank_right = fasta_sequences.get(chrom, "")[end:end+flank_size]
						
						data.append([seq, "exon", flank_left, flank_right, "Homo sapiens", chrom, transcript_id])
						
						# Criar íntron se houver próximo éxon
						if i < len(exons) - 1:
								next_start = exons[i + 1][1]
								intron_seq = fasta_sequences.get(chrom, "")[end:next_start]
								if intron_seq:
										flank_left = seq[-flank_size:]  # Últimos nucleotídeos do éxon atual
										flank_right = fasta_sequences.get(chrom, "")[next_start:next_start+flank_size]
										data.append([intron_seq, "intron", flank_left, flank_right, "Homo sapiens", chrom, transcript_id])
		
		return pd.DataFrame(data, columns=["sequence", "label", "flank_before", "flank_after", "organism", "chromosome", "Transcript_ID"])

# Arquivos de entrada
gtf_file = "datasets/gencode.v47.basic.annotation.gtf"
fasta_file = "datasets/GRCh38.primary_assembly.genome.fa"

# Carregar sequências do FASTA
fasta_sequences = load_fasta(fasta_file)

# Processar o GTF
df = process_gtf(gtf_file, fasta_sequences)

# Salvar o dataset
output_file = "introns_exons_dataset.csv"
df.to_csv(output_file, index=False)
print(f"Dataset salvo em {output_file}")
