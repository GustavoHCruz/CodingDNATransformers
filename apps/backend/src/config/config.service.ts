import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import * as YAML from 'yaml';
import {
  ConfigYaml,
  DataExtraction,
  FilesName,
  Paths,
} from './interfaces/config.interface';

@Injectable()
export class ConfigService {
  private readonly config: ConfigYaml;

  constructor() {
    const filePath = path.resolve(__dirname, '../../config.yaml');
    const file = fs.readFileSync(filePath, 'utf8');
    this.config = YAML.parse(file);
  }

  getConfig(): ConfigYaml {
    return this.config;
  }

  getPaths(): Paths {
    return this.config.paths;
  }

  getFilesName(): FilesName {
    return this.config.files_name;
  }

  getDataExtraction(): DataExtraction {
    return this.config.data_extraction;
  }
}
