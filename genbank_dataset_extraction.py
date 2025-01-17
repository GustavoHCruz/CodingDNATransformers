import csv

from Bio import SeqIO


def splicing_sites_extraction(genbank_file, csv_output_file):
  data = []

  with open(genbank_file, "r") as gb_file:
    for record in SeqIO.parse(gb_file, "genbank"):
      organism = record.annotations.get("organism", "")
      if (record.seq > 512):
        continue
      sequence = record.seq

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

  with open(csv_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["sequence", "label", "organism"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows()