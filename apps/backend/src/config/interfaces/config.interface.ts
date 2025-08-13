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
}

export interface DatasetGeneration {
  batch_size: number;
}

export interface ExInClassifier {
  gpt: number;
  bert: number;
  dnabert: number;
}

export interface TripletClassifier {
  bert: number;
  dnabert: number;
}

export interface DNATranslator {
  gpt: number;
}

export interface DatasetLengths {
  EXINCLASSIFIER: ExInClassifier;
  TRIPLETCLASSIFIER: TripletClassifier;
  DNATRANSLATOR: DNATranslator;
}

export interface ConfigYaml {
  paths: Paths;
  files_name: FilesName;
  data_extraction: DataExtraction;
  datasets_lengths: DatasetLengths;
  dataset_generation: DatasetGeneration;
}
