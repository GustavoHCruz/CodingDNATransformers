import pandas as pd

from genbank_dataset_extraction import splicing_sites_extraction

# splicing_sites_extraction("datasets/ExInSeqs.gb", "datasets/ExInSeqs_11M.csv")

df = pd.read_csv("datasets/ExInSeqs_11M.csv", keep_default_na=False)

seed = 1234

shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
df_exons = shuffled_df[shuffled_df["label"] == "exon"]
df_introns = shuffled_df[shuffled_df["label"] == "intron"]
df_exons_small = df_exons[df_exons["sequence"].str.len() < 128]
df_introns_small = df_introns[df_introns["sequence"].str.len() < 128]

def create_datasets(df_exons, df_introns, df_exons_small, df_introns_small, dataset_len, csv_name, create_small_version=True, datasets_dir="datasets"):
  exons = df_exons.sample(n=int(dataset_len/2), random_state=seed)
  introns = df_introns.sample(n=int(dataset_len/2), random_state=seed)
  df_exons = df_exons.drop(exons.index)
  df_introns = df_introns.drop(introns.index)
  
  df = pd.concat([exons, introns])
  df["flank_before"] = df["flank_before_extended"]
  df["flank_after"] = df["flank_after_extended"]
  df = df.drop(columns=["flank_before_extended", "flank_after_extended"])
  
  df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

  df.to_csv(f"{datasets_dir}/{csv_name}.csv", index=False)
  
  if create_small_version:
    exons_small = df_exons_small.sample(n=int(dataset_len/2), random_state=seed)
    introns_small = df_introns_small.sample(n=int(dataset_len/2), random_state=seed)
    df_exons_small = df_exons_small.drop(exons_small.index)
    df_introns_small = df_introns_small.drop(introns_small.index)
    
    df_small = pd.concat([exons_small, introns_small])
    df_small = df_small.drop(columns=["flank_before_extended", "flank_after_extended"])

    df_small = df_small.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_small.to_csv(f"{datasets_dir}/{csv_name}_small.csv", index=False)
  
  return df_exons, df_introns, df_exons_small, df_introns_small

dataset_sizes = [5000000, 100000, 30000, 5000]
dataset_names = ["ExInSeqs_5M", "ExInSeqs_100k", "ExInSeqs_30k", "ExInSeqs_5k"]

for size, name in zip(dataset_sizes, dataset_names):
  df_exons, df_introns, df_exons_small, df_introns_small = create_datasets(
    df_exons, df_introns, df_exons_small, df_introns_small, size, name
  )
