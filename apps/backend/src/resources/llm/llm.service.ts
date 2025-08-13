import { Injectable } from '@nestjs/common';
import { ParentRecordService } from '@resources/parent-record/parent-record.service';
import { SHARED_DIR } from 'common/constrants';
import { format } from 'fast-csv';
import { createWriteStream } from 'fs';
import * as path from 'path';

@Injectable()
export class LlmService {
  constructor(private readonly service: ParentRecordService) {}

  async generateCsvFromChildDataset(name: string, parentDatasetId: number) {
    const stream = format({ headers: true });

    const csvName = path.resolve(SHARED_DIR, 'temp', `${name}.csv`);
    const writable = createWriteStream(csvName);
    stream.pipe(writable);

    const data = await this.service.get(parentDatasetId);

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

    stream.end();
  }
}
