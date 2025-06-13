export interface Paths {
  raw_data: string;
  models: string;
  models_logs: string;
}

export interface GenbankFilesName {
  annotations: string;
}

export interface GencodeFileName {
  annotations: string;
  fasta: string;
}

export interface FilesName {
  genbank: GenbankFilesName;
  gencode: GencodeFileName;
}

export interface DataExtraction {
  save_batch_len: number;
  extraction_max_len: number;
}

export interface ConfigYaml {
  paths: Paths;
  files_name: FilesName;
  data_extraction: DataExtraction;
}
