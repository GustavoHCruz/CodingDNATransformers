import pandas as pd
from Bio import SeqIO


# Função para carregar a sequência do cromossomo a partir do FASTA
def load_fasta(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

# Função para processar o GTF e extrair íntrons e éxons
def process_gtf(gtf_file, fasta_sequences, flank_size=50):
    data = []
    transcripts = {}
    
    # Lendo o arquivo GTF
    with open(gtf_file, 'r') as gtf:
        for line in gtf:
            if line.startswith("#"):  # Ignorar comentários
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            chrom, source, feature, start, end, _, strand, _, attributes = fields
            
            if feature not in ["exon"]:
                continue
            
            start, end = int(start) - 1, int(end)  # Ajustando para índices Python
            
            # Extraindo o ID do transcrito
            transcript_id = "unknown"
            for attr in attributes.split("; "):
                if attr.startswith("transcript_id"):
                    transcript_id = attr.split('"')[1]
                    break
            
            if transcript_id not in transcripts:
                transcripts[transcript_id] = []
            transcripts[transcript_id].append((chrom, start, end, strand))
    
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
    
    return pd.DataFrame(data, columns=["Sequence", "Label", "Flanco_Anterior", "Flanco_Posterior", "Espécie", "Chromosome", "Transcript_ID"])

# Arquivos de entrada
gtf_file = "gencode.v47.annotation.gtf"
fasta_file = "gencode.v47.pc_transcripts.fa"

# Carregar sequências do FASTA
fasta_sequences = load_fasta(fasta_file)

# Processar o GTF
df = process_gtf(gtf_file, fasta_sequences)

# Salvar o dataset
output_file = "introns_exons_dataset.csv"
df.to_csv(output_file, index=False)
print(f"Dataset salvo em {output_file}")
