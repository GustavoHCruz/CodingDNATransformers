from Bio import SeqIO


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

def exin_translator_gc(gencode_fasta_file_path: str, gencode_annotations_file_path: str, parent_id: int, seq_max_len=512):
  record_counter = 0
  xd = []
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
        intron_seq = fasta_sequences.get(chrom, "")[end:next_start]

        if intron_seq:
          seqs.append(dict(
            sequence=str(reverse_complement(intron_seq, strand)),
            target="intron",
          ))
    
    final_seq = ""
    final_target = ""
    for seq in seqs:
      sequence, target = seq.values()
      final_seq += sequence
      final_target += f"({target}){sequence}({target})"

    if len(final_seq) < seq_max_len:		
      xd.append(dict(
        parent_id=parent_id,
        sequence=final_seq,
        target=final_target,
        organism="Homo sapiens"
      ))
      
  return xd

xd = exin_translator_gc("src/backend/data/raw/gencode/file1.fa", "src/backend/data/raw/gencode/file1.gtf", 1)