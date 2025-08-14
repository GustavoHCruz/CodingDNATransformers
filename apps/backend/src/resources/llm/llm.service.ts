import { ConfigService } from '@config/config.service';
import { Injectable } from '@nestjs/common';
import { ParentRecordService } from '@resources/parent-record/parent-record.service';
import { SHARED_DIR } from 'common/constrants';
import { format } from 'fast-csv';
import { createWriteStream } from 'fs';
import * as path from 'path';

@Injectable()
export class LlmService {
  constructor(
    private readonly parentRecordService: ParentRecordService,
    private readonly configService: ConfigService,
  ) {}

  async generateCsvFromChildDataset(name: string, parentDatasetId: number) {
    const batchSize = this.configService.getDatasetGeneration().batch_size;
    const stream = format({ headers: true });

    const csvName = path.resolve(SHARED_DIR, 'temp', `${name}.csv`);
    const writable = createWriteStream(csvName);
    stream.pipe(writable);

    let lastId: null | number = null;
    while (true) {
      const data = await this.parentRecordService.findByParentIdPaginated(
        parentDatasetId,
        lastId,
        batchSize,
      );

      if (!data.length) {
        break;
      }

      for (const row of data) {
        stream.write({
          sequence: row.sequence,
          target: row.target,
          organism: row.organism ?? '',
          gene: row.gene ?? '',
          flankBefore: row.flankBefore ?? '',
          flankAfter: row.flankAfter ?? '',
        });
      }

      lastId = data[data.length - 1].id;
    }

    stream.end();
  }
}
