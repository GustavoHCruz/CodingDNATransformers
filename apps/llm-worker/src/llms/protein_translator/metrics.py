import csv
import os
import subprocess
import tempfile

# Arquivos
csv_file = "resultados_proteinas.csv"
output_file = "resultados_com_blast.csv"

# Caminhos dos executáveis
blastp_path = "blastp"
makeblastdb_path = "makeblastdb"

# Ler CSV original
rows = []
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Criar CSV de saída com colunas extras
fieldnames = [
    "idx",
    "target",
    "pred",
    "edit_dist",
    "similarity",
    "blast_identity",
    "blast_score",
    "cov_target",  # % da sequência alvo coberta
    "cov_pred",    # % da sequência predita coberta
    "alignment"
]

with open(output_file, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(rows):
        target = row["target"].strip()
        pred = row["pred"].strip()

        # Defaults caso não haja sequências válidas
        blast_identity = ""
        blast_score = ""
        cov_target = ""
        cov_pred = ""
        alignment = ""

        if target and pred:
            # Criar arquivos temporários
            with tempfile.TemporaryDirectory() as tmpdir:
                target_fasta = os.path.join(tmpdir, "target.fasta")
                pred_fasta = os.path.join(tmpdir, "pred.fasta")
                db_name = os.path.join(tmpdir, "blastdb")

                # Salvar FASTAs
                with open(target_fasta, "w") as tf:
                    tf.write(f">target\n{target}\n")
                with open(pred_fasta, "w") as pf:
                    pf.write(f">pred\n{pred}\n")

                # Criar banco BLAST
                subprocess.run(
                    [makeblastdb_path, "-in", target_fasta, "-dbtype", "prot", "-out", db_name],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Rodar BLASTp. Incluí qlen e slen p/ calcular cobertura.
                # length = extensão do alinhamento local (inclui matches+mismatches+gaps)
                # pident = % identidade dentro desse alinhamento
                # qlen/slen = comprimentos totais das seqs query/subject no banco
                result = subprocess.run(
                    [
                        blastp_path,
                        "-query", pred_fasta,
                        "-db", db_name,
                        "-outfmt", "6 qseqid sseqid pident length score qlen slen"
                    ],
                    capture_output=True,
                    text=True,
                )

                out = result.stdout.strip()
                if out:
                    # BLAST pode retornar múltiplos hits; pegamos o 1º (melhor por default ordenação)
                    first_line = out.splitlines()[0]
                    cols = first_line.split("\t")
                    # Campos conforme ordem em -outfmt
                    # 0 qseqid, 1 sseqid, 2 pident, 3 length, 4 score, 5 qlen, 6 slen
                    blast_identity = cols[2]
                    aligned_len = int(cols[3])
                    blast_score = cols[4]
                    qlen = int(cols[5])
                    slen = int(cols[6])

                    # Cobertura alvo (subject)
                    cov_target = round(100.0 * aligned_len / slen, 2) if slen else 0.0
                    # Cobertura predita (query)
                    cov_pred = round(100.0 * aligned_len / qlen, 2) if qlen else 0.0

                    alignment = f"{cols[0]} vs {cols[1]} len={aligned_len}"
                else:
                    alignment = "No hit"

        # Atualiza linha e escreve
        row_out = {
            "idx": row["idx"],
            "target": target,
            "pred": pred,
            "edit_dist": row.get("edit_dist", ""),
            "similarity": row.get("similarity", ""),
            "blast_identity": blast_identity,
            "blast_score": blast_score,
            "cov_target": cov_target,
            "cov_pred": cov_pred,
            "alignment": alignment,
        }
        writer.writerow(row_out)

print(f"Resultados salvos em {output_file}")
