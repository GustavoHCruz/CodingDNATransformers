import os

import pandas as pd

from funcs.config_reading import read_datasets_configs
from genbank_dataset_extraction import (sequence_rebuild_extraction,
                                        sliding_window_extraction,
                                        splicing_sites_extraction)


def create_ExInSeqs_dataset():
  if not os.path.exists("datasets/ExInSeqs.csv"):
    if not os.path.exists("datasets/SplicingSitesSeqs.gb"):
      raise ValueError("SplicingSitesSeqs.gb not found in datasets directory.")
    splicing_sites_extraction("datasets/SplicingSitesSeqs.gb", "datasets/ExInSeqs.csv")
  df = pd.read_csv("datasets/ExInSeqs.csv", keep_default_na=False)

  seed = 1234

  datasets_config = read_datasets_configs("ExInSeqs")

  dataset_sizes = [i["len"] for i in datasets_config["sizes"]]
  dataset_names = [i["name"] for i in datasets_config["sizes"]]

  shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
  df_exons = shuffled_df[shuffled_df["label"] == "exon"]
  df_introns = shuffled_df[shuffled_df["label"] == "intron"]
  df_exons_small = df_exons[df_exons["sequence"].str.len() < datasets_config["version"]["small"]["sequence_len"]]
  df_introns_small = df_introns[df_introns["sequence"].str.len() < datasets_config["version"]["normal"]["sequence_len"]]

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

  for size, name in zip(dataset_sizes, dataset_names):
    df_exons, df_introns, df_exons_small, df_introns_small = create_datasets(
      df_exons, df_introns, df_exons_small, df_introns_small, size, name
    )

def create_RebuildSeqs_datasets():
  if not os.path.exists("datasets/RebuildSeqs.csv"):
    if not os.path.exists("datasets/SplicingSitesSeqs.gb"):
      raise ValueError("SplicingSitesSeqs.gb not found in datasets directory.")
    sequence_rebuild_extraction("datasets/SplicingSitesSeqs.gb", "datasets/RebuildSeqs.csv")
  df = pd.read_csv("datasets/RebuildSeqs.csv", keep_default_na=False)

  seed = 1234

  datasets_config = read_datasets_configs("RebuildSeqs")

  dataset_sizes = [i["len"] for i in datasets_config["sizes"]]
  dataset_names = [i["name"] for i in datasets_config["sizes"]]

  shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
  normal_df = shuffled_df[df["sequence"].str.len() < datasets_config["version"]["normal"]["sequence_len"]]
  small_df = shuffled_df[df["sequence"].str.len() < datasets_config["version"]["small"]["sequence_len"]]

  def create_datasets(normal_df, small_df, dataset_len, csv_name, create_small_version=True, datasets_dir="datasets"):
    normal = normal_df.sample(n=int(dataset_len), random_state=seed)
    normal_df = normal_df.drop(normal.index)
    
    normal = normal.sample(frac=1, random_state=seed).reset_index(drop=True)

    normal.to_csv(f"{datasets_dir}/{csv_name}.csv", index=False)
    
    if create_small_version:
      small = small_df.sample(n=int(dataset_len), random_state=seed)
      small_df = small_df.drop(small.index)

      small = small.sample(frac=1, random_state=seed).reset_index(drop=True)

      small.to_csv(f"{datasets_dir}/{csv_name}_small.csv", index=False)
    
    return normal_df, small_df

  for size, name in zip(dataset_sizes, dataset_names):
    normal_df, small_df = create_datasets(normal_df, small_df, size, name)

create_RebuildSeqs_datasets()

def create_SWExInSeqs_datasets():
  if not os.path.exists("datasets/SWExInSeqs.csv"):
    if not os.path.exists("datasets/SplicingSitesSeqs.gb"):
      raise ValueError("SplicingSitesSeqs.gb not found in datasets directory.")
    sliding_window_extraction("datasets/SplicingSitesSeqs.gb", "datasets/SWExInSeqs.csv")
  df = pd.read_csv("datasets/SWExInSeqs.csv", keep_default_na=False)

  seed = 1234

  datasets_config = read_datasets_configs("SWExInSeqs")

  dataset_sizes = [i["len"] for i in datasets_config["sizes"]]
  dataset_names = [i["name"] for i in datasets_config["sizes"]]

  df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

  def create_datasets(original_df, dataset_len, csv_name, datasets_dir="datasets"):
    new_df = original_df.sample(n=int(dataset_len), random_state=seed)
    original_df = original_df.drop(new_df.index)
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    new_df.to_csv(f"{datasets_dir}/{csv_name}.csv", index=False)

    return original_df

  for size, name in zip(dataset_sizes, dataset_names):
    print(f"Generating dataset {name} of size {size}")
    df = create_datasets(df, size, name)