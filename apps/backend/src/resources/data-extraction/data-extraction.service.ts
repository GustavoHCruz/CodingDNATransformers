import { ExtractionGrpcClientService } from '@grpc/extraction.grpc-client.service';
import { ExtractionService } from '@grpc/interfaces/extraction.interface';
import { Injectable } from '@nestjs/common';
import { ApproachEnum, OriginEnum, ProgressTypeEnum } from '@prisma/client';
import { ParentDatasetService } from '@resources/parent-dataset/parent-dataset.service';
import { CreateParentRecordDto } from '@resources/parent-record/dto/create-parent-record.dto';
import { ParentRecordService } from '@resources/parent-record/parent-record.service';
import { ProgressTrackerService } from '@resources/progress-tracker/progress-tracker.service';
import { RawFileInfoService } from '@resources/raw-file-info/raw-file-info.service';
import { ConfigService } from 'config/config.service';
import { observableToAsyncIterable } from 'utils/observable-to-async';
import { v4 as uuidv4 } from 'uuid';
import {
  CreateDataExtractionDto,
  CreateDataExtractionResponseDto,
} from './dto/data-extraction.dto';

type ProcessTask = {
  extractor: keyof ExtractionService;
  annotationsPath: string;
  fastaPath?: string;
  parentId: number;
  taskId: number;
  sequenceMaxLength: number;
  totalRecords: number;
  batchSize: number;
  approach: ApproachEnum;
};

type ExtractionTask = {
  extractor: keyof ExtractionService;
  annotationsPath: string;
  approach: ApproachEnum;
  origin: OriginEnum;
  fastaPath?: string;
};

@Injectable()
export class DataExtractionService {
  constructor(
    private parentDatasetService: ParentDatasetService,
    private parentRecordService: ParentRecordService,
    private progressTrackerService: ProgressTrackerService,
    private rawFileInfoService: RawFileInfoService,
    private configYaml: ConfigService,
    private extractionService: ExtractionGrpcClientService,
  ) {}

  async initialConfiguration(path: string, approach: ApproachEnum) {
    const fileInfo = await this.rawFileInfoService.findByFileNameAndApproach(
      path,
      approach,
    );

    let totalRecords = 0;
    let progressType: ProgressTypeEnum = ProgressTypeEnum.COUNTER;

    if (fileInfo) {
      totalRecords = fileInfo.totalRecords;
      progressType = ProgressTypeEnum.PERCENTAGE;
    }

    return { totalRecords, progressType };
  }

  async processTask({
    extractor,
    parentId,
    sequenceMaxLength,
    annotationsPath,
    fastaPath,
    taskId,
    totalRecords,
    batchSize,
    approach,
  }: ProcessTask) {
    let batch: CreateParentRecordDto[] = [];
    let recordCount = 0;
    let raw_total = 0;

    const observable = this.extractionService.call(extractor, {
      sequenceMaxLength,
      annotationsPath,
      fastaPath,
    });
    const asyncIterable = observableToAsyncIterable(observable);

    try {
      for await (const record of asyncIterable) {
        batch.push({ ...record, parentDatasetId: parentId });
        raw_total++;

        if (batch.length >= batchSize) {
          const filtered = this.parentRecordService.removeDuplicated(
            batch,
            taskId,
          );
          recordCount += filtered.length;
          await this.parentRecordService.createMany(filtered);
          await this.progressTrackerService.postProgress(
            taskId,
            raw_total,
            totalRecords,
          );

          batch = [];
        }
      }

      const filtered = this.parentRecordService.removeDuplicated(batch, taskId);
      recordCount += filtered.length;
      await this.parentRecordService.createMany(filtered);
      await this.progressTrackerService.postProgress(
        taskId,
        raw_total,
        totalRecords,
      );
    } catch {
      await this.progressTrackerService.finish(taskId, false);
    } finally {
      await this.progressTrackerService.finish(taskId);
      this.parentRecordService.clearTaskState(taskId);
    }
    await this.parentDatasetService.update(parentId, {
      recordCount,
    });

    if (!totalRecords) {
      await this.rawFileInfoService.create({
        approach,
        fileName: annotationsPath,
        totalRecords: recordCount,
      });
    }
  }

  async extractionTask({
    extractor,
    annotationsPath,
    approach,
    origin,
    fastaPath,
  }: ExtractionTask): Promise<number> {
    const batchSize = this.configYaml.getDataExtraction().save_batch_len;
    const sequenceMaxLength =
      this.configYaml.getDataExtraction().extraction_max_len;
    const { totalRecords, progressType } = await this.initialConfiguration(
      annotationsPath,
      approach,
    );

    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.-]/g, '');
    const parentId = (
      await this.parentDatasetService.create({
        name: `approach${approach}-${timestamp}-${uuidv4()}`,
        approach,
        origin,
      })
    ).id;

    const taskId = (
      await this.progressTrackerService.create({
        progressType,
        taskName: `origin:${origin}-approach:${approach}-parent:${parentId}`,
      })
    ).id;

    this.processTask({
      annotationsPath,
      extractor,
      parentId,
      sequenceMaxLength,
      taskId,
      fastaPath,
      batchSize,
      totalRecords,
      approach,
    });

    return taskId;
  }

  async extract(
    data: CreateDataExtractionDto,
  ): Promise<CreateDataExtractionResponseDto> {
    const response = new CreateDataExtractionResponseDto();

    const genbank = data.genbank;
    const gencode = data.gencode;

    if (genbank) {
      const origin = OriginEnum.GENBANK;
      const annotationsPath = `${this.configYaml.getPaths().raw_data}/${this.configYaml.getFilesName().genbank.annotations}`;

      if (genbank.ExInClassifier) {
        const approach = ApproachEnum.EXINCLASSIFIER;
        const taskId = await this.extractionTask({
          extractor: 'ExInClassifierGenbank',
          annotationsPath,
          origin,
          approach,
        });
        response.genbank.ExInClassifier = taskId;
      }

      if (genbank.ExInTranslator) {
        const approach = ApproachEnum.EXINTRANSLATOR;
        const taskId = await this.extractionTask({
          extractor: 'ExInTranslatorGenbank',
          annotationsPath,
          origin,
          approach,
        });
        response.genbank.ExInTranslator = taskId;
      }

      if (genbank.SlidingWindowTagger) {
        const approach = ApproachEnum.SLIDINGWINDOWEXTRACTION;
        const taskId = await this.extractionTask({
          extractor: 'SlidingWindowTaggerGenbank',
          annotationsPath,
          origin,
          approach,
        });
        response.genbank.SlidingWindowTagger = taskId;
      }

      if (genbank.ProteinTranslator) {
        const approach = ApproachEnum.PROTEINTRANSLATOR;
        const taskId = await this.extractionTask({
          extractor: 'ProteinTranslatorGenbank',
          annotationsPath,
          origin,
          approach,
        });
        response.genbank.ProteinTranslator = taskId;
      }
    }

    if (gencode) {
      const origin = OriginEnum.GENCODE;
      const annotationsPath = `${this.configYaml.getPaths().raw_data}/${this.configYaml.getFilesName().gencode.annotations}`;
      const fastaPath = `${this.configYaml.getPaths().raw_data}/${this.configYaml.getFilesName().gencode.fasta}`;

      if (gencode.ExInClassifier) {
        const approach = ApproachEnum.EXINCLASSIFIER;
        const taskId = await this.extractionTask({
          extractor: 'ExInClassifierGencode',
          annotationsPath,
          origin,
          approach,
          fastaPath,
        });
        response.gencode.ExInClassifier = taskId;
      }

      if (gencode.ExInTranslator) {
        const approach = ApproachEnum.EXINTRANSLATOR;
        const taskId = await this.extractionTask({
          extractor: 'ExInTranslatorGencode',
          annotationsPath,
          origin,
          approach,
          fastaPath,
        });
        response.gencode.ExInTranslator = taskId;
      }

      if (gencode.SlidingWindowTagger) {
        const approach = ApproachEnum.SLIDINGWINDOWEXTRACTION;
        const taskId = await this.extractionTask({
          extractor: 'SlidingWindowTaggerGencode',
          annotationsPath,
          origin,
          approach,
          fastaPath,
        });
        response.gencode.SlidingWindowTagger = taskId;
      }

      if (gencode.ProteinTranslator) {
        const approach = ApproachEnum.PROTEINTRANSLATOR;
        const taskId = await this.extractionTask({
          extractor: 'ProteinTranslatorGencode',
          annotationsPath,
          origin,
          approach,
          fastaPath,
        });
        response.gencode.ProteinTranslator = taskId;
      }
    }

    return response;
  }
}
