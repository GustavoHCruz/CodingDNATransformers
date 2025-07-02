import { Injectable } from '@nestjs/common';
import { ChildRecordService } from '@resources/child-record/child-record.service';
import { SHARED_DIR } from 'common/constrants';
import { format } from 'fast-csv';
import { createWriteStream } from 'fs';
import path from 'path';

const ALLOWED_MODELS: Record<string, (modelName: string) => boolean> = {
  EXINCLASSIFIER: (name) =>
    ['gpt2', 'bert-base-uncased', 'dnabert'].includes(name),
  EXINTRANSLATOR: (name) => ['gpt2', 't5-base'].includes(name),
  SLIDINGWINDOWEXTRACTION: (name) =>
    ['gpt2', 'bert-base-uncased', 'dnabert'].includes(name),
  PROTEINTRANSLATOR: (name) => ['gpt2', 't5-base'].includes(name),
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
}
