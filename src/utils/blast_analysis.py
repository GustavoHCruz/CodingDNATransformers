import os
import subprocess
import tempfile


def blast_analysis(
	pred: str,
	target: str,
	blastp_path: str = "blastp",
	makeblast_path: str = "makeblastdb"
) -> dict:
	blast_identity = ""
	cov_target = ""
	cov_pred = ""
	alignment = ""

	if pred.strip() and target.strip():
		with tempfile.TemporaryDirectory() as tmpdir:
			target_fasta = os.path.join(tmpdir, "target.fasta")
			pred_fasta = os.path.join(tmpdir, "pred.fasta")
			db_name = os.path.join(tmpdir, "blastdb")

			with open(target_fasta, "w") as tf:
				tf.write(f">target\n{target}\n")
			with open(pred_fasta, "w") as pf:
				pf.write(f">pred\n{pred}\n")

			subprocess.run(
				[makeblast_path, "-in", target_fasta, "-dbtype", "prot", "-out", db_name],
				check=True,
				stdout=subprocess.DEVNULL,
				stderr=subprocess.DEVNULL
			)

			result = subprocess.run(
				[
					blastp_path,
					"-query", pred_fasta,
					"-db", db_name,
					"-outfmt", "6 qseqid sseqid pident length score qlen slen"
				],
				capture_output=True,
				text=True
			)

			out = result.stdout.strip()
			if out:
				first_line = out.splitlines()[0]
				cols = first_line.split("\t")

				blast_identity = round(float(cols[2]) / 100.0, 4)
				aligned_len = int(cols[3])
				qlen = int(cols[5])
				slen = int(cols[6])

				cov_target = aligned_len / slen if slen else 0.0
				cov_pred   = aligned_len / qlen if qlen else 0.0

				alignment = f"{cols[0]} vs {cols[1]} len={aligned_len}"
			else:
				alignment = "No hit"

	return {
		"blast_identity": blast_identity,
		"cov_target": cov_target,
		"cov_pred": cov_pred,
		"alignment": alignment
	}