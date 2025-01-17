import csv

from Bio import SeqIO

try:
	from IPython import get_ipython
	in_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except ImportError:
	in_notebook = False

if in_notebook:
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm

def splicing_sites_extraction(genbank_file, csv_output_file):
	with open(genbank_file, "r") as gb_file:
		total_records = sum(1 for _ in SeqIO.parse(gb_file, "genbank"))
	
	data = []
	progress_bar = tqdm(total=total_records, desc="File Reading Progress", position=0, leave=True)

	with open(genbank_file, "r") as gb_file:
		for record in SeqIO.parse(gb_file, "genbank"):
			if (len(record.seq) > 512):
				progress_bar.update(1)
				continue

			sequence = record.seq
			organism = record.annotations.get("organism", "")

			for feature in record.features:
				if feature.type in ["intron", "exon"]:
					location = feature.location
					feature_sequence = sequence[location.start:location.end]
					label = feature.type

					data.append({
						"sequence": str(feature_sequence),
						"label": label,
						"organism": organism,
					})

			progress_bar.update(1)

	with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
		fieldnames = ["sequence", "label", "organism"]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		writer.writerows(data)