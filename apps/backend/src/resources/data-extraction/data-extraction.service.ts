import { ExtractionGrpcClientService } from '@grpc/data-extraction/extraction.grpc-client.service';
import { ExtractionResponse } from '@grpc/data-extraction/interfaces/extraction.interface';
import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { OriginEnum, ProgressTypeEnum } from '@prisma/client';
import { DnaSequenceService } from '@resources/dna-sequence/dna-sequence.service';

import { FeatureSequenceService } from '@resources/feature-sequence/feature-sequence.service';
import { ProgressTrackerService } from '@resources/progress-tracker/progress-tracker.service';
import { RawFileInfoService } from '@resources/raw-file-info/raw-file-info.service';
import { ConfigService } from 'config/config.service';
import { observableToAsyncIterable } from 'utils/observable-to-async';
import { DataExtractionReturn } from './dto/approachs.dto';

@Injectable()
export class DataExtractionService {
  constructor(
    private progressTrackerService: ProgressTrackerService,
    private rawFileInfoService: RawFileInfoService,
    private configYaml: ConfigService,
    private extractionService: ExtractionGrpcClientService,
    private dnaSequenceService: DnaSequenceService,
    private featureSequenceService: FeatureSequenceService,
  ) {}

  async initialConfiguration(path: string, origin: OriginEnum) {
    const fileInfo = await this.rawFileInfoService.findByFileNameAndApproach(
      path,
      origin,
    );

    let totalRecords = 0;
    let progressType: ProgressTypeEnum = ProgressTypeEnum.COUNTER;

    if (fileInfo) {
      totalRecords = fileInfo.totalRecords;
      progressType = ProgressTypeEnum.PERCENTAGE;
    }

    return { totalRecords, progressType };
  }

  async extractorFn(
    annotationsPath: string,
    taskId: number,
    origin: OriginEnum,
    batchSize: number,
    totalRecords?: number,
  ) {
    try {
      let batch: ExtractionResponse[] = [];
      let recordCount = 0;

      const asyncIterable = observableToAsyncIterable(
        this.extractionService.callExtract({ path: annotationsPath }),
      );

      const processBatch = async (records: ExtractionResponse[]) => {
        const cdsAll: any[] = [];
        const exinAll: any[] = [];

        for (const item of records) {
          const { cds, exin, ...dna } = item;
          const dnaSequenceId = (await this.dnaSequenceService.create(dna)).id;

          if (cds?.length) {
            cdsAll.push(...cds.map((e) => ({ ...e, dnaSequenceId })));
          }
          if (exin?.length) {
            exinAll.push(...exin.map((e) => ({ ...e, dnaSequenceId })));
          }
        }

        if (cdsAll.length) {
          await this.featureSequenceService.createMany(cdsAll);
        }
        if (exinAll.length) {
          await this.featureSequenceService.createMany(exinAll);
        }
      };

      for await (const record of asyncIterable) {
        batch.push(record);

        if (batch.length >= batchSize) {
          await processBatch(batch);
          recordCount += batch.length;
          batch = [];

          await this.progressTrackerService.postProgress(
            taskId,
            recordCount,
            totalRecords,
          );
        }
      }

      if (batch.length > 0) {
        await processBatch(batch);
        recordCount += batch.length;
      }

      await this.progressTrackerService.postProgress(
        taskId,
        recordCount,
        totalRecords,
      );

      if (!totalRecords) {
        await this.rawFileInfoService.create({
          origin,
          totalRecords: recordCount,
          fileName: annotationsPath,
        });
      }

      await this.progressTrackerService.finish(taskId);
    } catch (error) {
      await this.progressTrackerService.finish(taskId, false);
      throw new InternalServerErrorException(error);
    }
  }

  async extract(): Promise<DataExtractionReturn> {
    const origin = OriginEnum.GENBANK;
    const annotationsPath = `${this.configYaml.getPaths().raw_data}/${this.configYaml.getFilesName().genbank.annotations}`;
    const batchSize = this.configYaml.getDataExtraction().save_batch_len;

    const { totalRecords, progressType } = await this.initialConfiguration(
      annotationsPath,
      origin,
    );

    const taskId = (
      await this.progressTrackerService.create({
        progressType,
      })
    ).id;

    this.extractorFn(annotationsPath, taskId, origin, batchSize, totalRecords);

    return { taskId };
  }
}
