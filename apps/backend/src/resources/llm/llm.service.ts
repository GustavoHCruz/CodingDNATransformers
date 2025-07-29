import { Injectable } from '@nestjs/common';
import { ChildRecordService } from '@resources/child-record/child-record.service';
import { SHARED_DIR } from 'common/constrants';
import { format } from 'fast-csv';
import { createWriteStream } from 'fs';
import * as path from 'path';
import { CreateModelDto } from './dto/createModel.dto';

const GPTS_MODELS = ['gpt2', 'gpt2-medium'];

const ALLOWED_MODELS: Record<string, (modelName: string) => boolean> = {
  EXINCLASSIFIER: (name) =>
    [...GPTS_MODELS, 'bert-base-uncased', 'dnabert'].includes(name),
  SLIDINGWINDOWEXTRACTION: (name) =>
    [...GPTS_MODELS, 'bert-base-uncased', 'dnabert'].includes(name),
  PROTEINTRANSLATOR: (name) => [...GPTS_MODELS, 't5-base'].includes(name),
};

@Injectable()
export class LlmService {
  constructor(private readonly childRecordService: ChildRecordService) {}

  async generateCsvFromChildDataset(
    executionUuid: string,
    childDatasetId: number,
  ) {
    const stream = format({ headers: true });

    const csvName = path.resolve(SHARED_DIR, 'temp', `${executionUuid}.csv`);
    const writable = createWriteStream(csvName);
    stream.pipe(writable);

    const trainRecordGenerator =
      this.childRecordService.streamFindAllByChildDatasetId(childDatasetId);

    for await (const record of trainRecordGenerator) {
      stream.write(record);
    }

    stream.end();
  }

  async createModel(data: CreateModelDto) {
    const { approach, checkpointName, modelAlias, parentId } = data;
  }
}
