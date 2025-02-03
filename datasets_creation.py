import pandas as pd

from genbank_dataset_extraction import splicing_sites_extraction

splicing_sites_extraction("datasets/ExInSeqs.gb", "datasets/ExinSeqs_11M.csv")

df = pd.read_csv("datasets/ExInSeqs_11M.csv", keep_default_na=False)

shuffled_df = df.sample(frac=1).reset_index(drop=True)

df_exons = shuffled_df[shuffled_df["label"] == "exon"]
df_introns = shuffled_df[shuffled_df["label"] == "intron"]

df_exons_small = df_exons[df_exons["sequence"].str.len() < 128]
df_introns_small = df_introns[df_introns["sequence"].str.len() < 128]

print(len(df_exons))
print(len(df_exons_small))
print(len(df_introns))
print(len(df_introns_small))

df_3k_small = pd.concat([df_exons_small.sample(n=1500), df_introns_small.sample(n=1500)])
df_3k_small = df_3k_small.drop(columns=["flank_before_extended", "flank_after_extended"])

df_3k_small = df_3k_small.sample(frac=1).reset_index(drop=True)
print(f"Exons: {len(df_3k_small[df_3k_small["label"] == "exon"])}")
print(f"Introns: {len(df_3k_small[df_3k_small["label"] == "intron"])}")
print(f"Total Len: {len(df_3k_small)}")

df_3k_small.to_csv("datasets/ExInSeqs_3k_small.csv", index=False)

df_3k = pd.concat([df_exons.sample(n=1500), df_introns.sample(n=1500)])
df_3k["flank_before"] = df_3k["flank_before_extended"]
df_3k["flank_after"] = df_3k["flank_after_extended"]
df_3k = df_3k.drop(columns=["flank_before_extended", "flank_after_extended"])

df_3k = df_3k.sample(frac=1).reset_index(drop=True)
print(f"Exons: {len(df_3k[df_3k["label"] == "exon"])}")
print(f"Introns: {len(df_3k[df_3k["label"] == "intron"])}")
print(f"Total Len: {len(df_3k)}")

df_3k.to_csv("datasets/ExInSeqs_3k.csv", index=False)

df_30k_small = pd.concat([df_exons_small.sample(n=15000), df_introns_small.sample(n=15000)])
df_30k_small = df_30k_small.drop(columns=["flank_before_extended", "flank_after_extended"])

df_30k_small = df_30k_small.sample(frac=1).reset_index(drop=True)
print(f"Exons: {len(df_30k_small[df_30k_small["label"] == "exon"])}")
print(f"Introns: {len(df_30k_small[df_30k_small["label"] == "intron"])}")
print(f"Total Len: {len(df_30k_small)}")

df_30k_small.to_csv("datasets/ExInSeqs_30k_small.csv", index=False)

df_30k = pd.concat([df_exons.sample(n=15000), df_introns.sample(n=15000)])
df_30k["flank_before"] = df_30k["flank_before_extended"]
df_30k["flank_after"] = df_30k["flank_after_extended"]
df_30k = df_30k.drop(columns=["flank_before_extended", "flank_after_extended"])

df_30k = df_30k.sample(frac=1).reset_index(drop=True)
print(f"Exons: {len(df_30k[df_30k["label"] == "exon"])}")
print(f"Introns: {len(df_30k[df_30k["label"] == "intron"])}")
print(f"Total Len: {len(df_30k)}")

df_30k.to_csv("datasets/ExInSeqs_30k.csv", index=False)

df_100k_small = pd.concat([df_exons_small.sample(n=50000), df_introns_small.sample(n=50000)])
df_100k_small = df_100k_small.drop(columns=["flank_before_extended", "flank_after_extended"])

df_100k_small = df_100k_small.sample(frac=1).reset_index(drop=True)
print(f"Exons: {len(df_100k_small[df_100k_small["label"] == "exon"])}")
print(f"Introns: {len(df_100k_small[df_100k_small["label"] == "intron"])}")
print(f"Total Len: {len(df_100k_small)}")

df_100k_small.to_csv("datasets/ExInSeqs_100k_small.csv", index=False)

df_100k = pd.concat([df_exons.sample(n=50000), df_introns.sample(n=50000)])
df_100k["flank_before"] = df_100k["flank_before_extended"]
df_100k["flank_after"] = df_100k["flank_after_extended"]
df_100k = df_100k.drop(columns=["flank_before_extended", "flank_after_extended"])

df_100k = df_100k.sample(frac=1).reset_index(drop=True)
print(f"Exons: {len(df_100k[df_100k["label"] == "exon"])}")
print(f"Introns: {len(df_100k[df_100k["label"] == "intron"])}")
print(f"Total Len: {len(df_100k)}")

df_100k.to_csv("datasets/ExInSeqs_100k.csv", index=False)