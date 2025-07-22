import csv
import os
import subprocess
import tempfile

# Caminho do CSV
csv_file = "resultados_proteinas.csv"
output_file = "resultados_com_blast.csv"

# Caminho do BLAST executável (ajuste se necessário)
blastp_path = "blastp"
makeblastdb_path = "makeblastdb"

# Ler CSV original
rows = []
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Criar CSV de saída com colunas adicionais
with open(output_file, "w", newline="", encoding="utf-8") as f_out:
    fieldnames = ["idx", "target", "pred", "edit_dist", "similarity", "blast_identity", "blast_score", "alignment"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows):
        target = row["target"]
        pred = row["pred"]

        # Ignorar entradas vazias
        if not target or not pred:
            row.update({"blast_identity": "", "blast_score": "", "alignment": ""})
            writer.writerow(row)
            continue

        # Criar arquivos temporários
        with tempfile.TemporaryDirectory() as tmpdir:
            target_fasta = os.path.join(tmpdir, "target.fasta")
            pred_fasta = os.path.join(tmpdir, "pred.fasta")
            db_name = os.path.join(tmpdir, "blastdb")

            # Salvar target e pred em FASTA
            with open(target_fasta, "w") as tf:
                tf.write(f">target\n{target}\n")
            with open(pred_fasta, "w") as pf:
                pf.write(f">pred\n{pred}\n")

            # Criar banco BLAST do target
            subprocess.run([makeblastdb_path, "-in", target_fasta, "-dbtype", "prot", "-out", db_name],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Rodar blastp com pred vs target
            result = subprocess.run(
                [blastp_path, "-query", pred_fasta, "-db", db_name, "-outfmt", "6 qseqid sseqid pident length score"],
                capture_output=True, text=True
            )

            if result.stdout.strip():
                cols = result.stdout.strip().split("\t")
                blast_identity = cols[2]  # identidade (%)
                blast_score = cols[4]    # score
                alignment = f"{cols[0]} vs {cols[1]} len={cols[3]}"
            else:
                blast_identity = ""
                blast_score = ""
                alignment = "No hit"

            # Adicionar resultados ao row
            row.update({
                "blast_identity": blast_identity,
                "blast_score": blast_score,
                "alignment": alignment
            })
            writer.writerow(row)

print(f"Resultados salvos em {output_file}")
