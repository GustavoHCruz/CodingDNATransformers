import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import * as YAML from 'yaml';
import { ConfigYaml, Directories, Llms } from './interfaces/config.interface';

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

  getDirectories(): Directories {
    return this.config.directories;
  }

  getLlms(): Llms {
    return this.config.llms;
  }
}
