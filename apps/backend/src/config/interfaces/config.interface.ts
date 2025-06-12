export interface Directories {
  raw_data: string;
  models: string;
  models_logs: string;
}

export interface Llms {
  sequenceMaxLength: number;
}

export interface ConfigYaml {
  directories: Directories;
  llms: Llms;
}
