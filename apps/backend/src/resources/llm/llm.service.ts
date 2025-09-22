import { ConfigService } from '@config/config.service';
import { Injectable } from '@nestjs/common';
import { ChildRecordService } from '@resources/child-record/child-record.service';
import { ParentRecordService } from '@resources/parent-record/parent-record.service';
import { STORAGE_DIR } from 'common/constrants';
import { CsvFormatterStream, format } from 'fast-csv';
import { createWriteStream } from 'fs';
import * as path from 'path';
import { CreateCsvFromParentDataset, DatasetType } from './dto/createModel.dto';

@Injectable()
export class LlmService {
  constructor(
    private readonly parentRecordService: ParentRecordService,
    private readonly childRecordService: ChildRecordService,
    private readonly configService: ConfigService,
  ) {}

  async recordsFromParentId(
    fileStream: CsvFormatterStream<any, any>,
    parentId: number,
    batchSize: number,
  ) {
    let lastId: number | null = null;

    while (true) {
      const records = await this.parentRecordService.findByParentIdPaginated(
        parentId,
        lastId,
        batchSize,
      );

      if (!records.length) break;

      for (const row of records) {
        fileStream.write({
          sequence: row.sequence,
          target: row.target,
          organism: row.organism ?? '',
          gene: row.gene ?? '',
          flankBefore: row.flankBefore ?? '',
          flankAfter: row.flankAfter ?? '',
        });
      }

      lastId = records[records.length - 1].id;
    }

    fileStream.end();
  }

  async recordsFromChildId(
    fileStream: CsvFormatterStream<any, any>,
    childId: number,
    batchSize: number,
  ) {
    let lastId: number | null = null;

    while (true) {
      const records = await this.childRecordService.findByChildIdPaginated(
        childId,
        lastId,
        batchSize,
      );

      if (!records.length) break;

      for (const row of records) {
        const parentRecord = row.parentRecord;
        if (parentRecord) {
          fileStream.write({
            sequence: parentRecord.sequence,
            target: parentRecord.target,
            organism: parentRecord.organism ?? '',
            gene: parentRecord.gene ?? '',
            flankBefore: parentRecord.flankBefore ?? '',
            flankAfter: parentRecord.flankAfter ?? '',
          });
        }
      }

      lastId = records[records.length - 1].id;
    }

    fileStream.end();
  }

  async generateCsv(data: CreateCsvFromParentDataset) {
    const batchSize = this.configService.getDatasetGeneration().batch_size;
    const stream = format({ headers: true });

    const csvName = path.resolve(STORAGE_DIR, 'data', `${data.fileAlias}.csv`);
    const writable = createWriteStream(csvName);
    stream.pipe(writable);

    if (data.type === DatasetType.Parent) {
      await this.recordsFromParentId(stream, data.datasetId, batchSize);
    } else {
      await this.recordsFromChildId(stream, data.datasetId, batchSize);
    }
  }
}
