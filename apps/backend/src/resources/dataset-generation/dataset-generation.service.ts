import { ConfigService } from '@config/config.service';
import { ForbiddenException, Injectable } from '@nestjs/common';
import {
  ApproachEnum,
  ModelTypeEnum,
  OriginEnum,
  ProgressTypeEnum,
} from '@prisma/client';
import { ChildDatasetService } from '@resources/child-dataset/child-dataset.service';
import { ChildRecordService } from '@resources/child-record/child-record.service';
import { FeatureSequenceService } from '@resources/feature-sequence/feature-sequence.service';
import { GenerationBatchService } from '@resources/generation-batch/generation-batch.service';
import { ParentDatasetService } from '@resources/parent-dataset/parent-dataset.service';
import { ParentRecordService } from '@resources/parent-record/parent-record.service';
import { ProgressTrackerService } from '@resources/progress-tracker/progress-tracker.service';
import {
  CreateProcessedDatasetsDto,
  CreateProcessedDatasetsDtoResponse,
} from './dto/create-processed-datasets.dto';
import {
  CreateRawDatasetsDto,
  CreateRawDatasetsDtoResponse,
} from './dto/create-raw-datasets.dto';

@Injectable()
export class DatasetGenerationService {
  constructor(
    private readonly progressService: ProgressTrackerService,
    private readonly childDatasetService: ChildDatasetService,
    private readonly childRecordService: ChildRecordService,
    private readonly generationBatchService: GenerationBatchService,
    private readonly configService: ConfigService,
    private readonly parentRecordService: ParentRecordService,
    private readonly parentDatasetService: ParentDatasetService,
    private readonly FeatureSequenceService: FeatureSequenceService,
  ) {}

  private async backgroundChildDatasetCreation(
    childDatasetId: number,
    taskId: number,
    totalRecords: number,
    parentRecordIds: number[],
  ): Promise<number> {
    const batchSize =
      this.configService.getDatasetGeneration().batch_size || 100;

    (async () => {
      let counter = 0;

      for (let i = 0; i < parentRecordIds.length; i += batchSize) {
        const batch = parentRecordIds.slice(i, i + batchSize);

        const data = batch.map((parentId) => ({
          childDatasetId: childDatasetId,
          parentRecordId: parentId,
        }));

        await this.childRecordService.createMany(data);

        counter += batch.length;
        await this.progressService.postProgress(taskId, counter, totalRecords);
      }

      await this.progressService.finish(taskId);
    })();

    return taskId;
  }

  private seededRandom(seed: number): () => number {
    let x = Math.sin(seed) * 10000;
    return () => {
      x = Math.sin(x) * 10000;
      return x - Math.floor(x);
    };
  }

  async generateProcessedDatasets(
    data: CreateProcessedDatasetsDto,
  ): Promise<CreateProcessedDatasetsDtoResponse[]> {
    const responses: CreateProcessedDatasetsDtoResponse[] = [];

    const totalAvailable = await this.parentRecordService.countByApproach(
      data.approach,
    );
    const requestedAmount = data.datasets.reduce(
      (acc, dataset) => acc + dataset.size,
      0,
    );
    if (requestedAmount > totalAvailable) {
      throw new ForbiddenException(
        'Cannot create the datasets, the requested amount is higher than the records on database',
      );
    }

    const batchName = data.batchName || `batch-${new Date().toISOString()}`;
    const generationBatch = await this.generationBatchService.create({
      name: batchName,
    });

    if (!generationBatch?.id) {
      throw new Error('Could not retrieve Generation Batch');
    }

    const generationBatchId = generationBatch.id;
    const totalToCreate = data.datasets.reduce((acc, cur) => acc + cur.size, 0);

    const parentRecordIds = await this.parentRecordService.findByApproach(
      data.approach,
      totalToCreate,
    );

    const rng = this.seededRandom(data.seed || 1234);
    const shuffledParentIds = [...parentRecordIds].sort(() => rng() - 0.5);

    let recordsSplitStart = 0;

    for (const child of data.datasets) {
      const recordsSplitEnd = recordsSplitStart + child.size;
      const parentIdsToUse = shuffledParentIds
        .slice(recordsSplitStart, recordsSplitEnd)
        .map((parent) => parent.id);

      const newChildDataset = await this.childDatasetService.create({
        name: child.name,
        approach: data.approach,
        modelType: data.modelType,
        batchId: generationBatchId,
        recordCount: child.size,
      });

      const task = await this.progressService.create({
        progressType: ProgressTypeEnum.PERCENTAGE,
        taskName: `batch:${generationBatchId} child:${newChildDataset.id}`,
      });

      this.backgroundChildDatasetCreation(
        newChildDataset.id,
        task.id,
        child.size,
        parentIdsToUse,
      );

      responses.push({
        name: child.name,
        taskId: task.id,
      });

      recordsSplitStart = recordsSplitEnd;
    }

    return responses;
  }

  async generateRawDatasets(
    data: CreateRawDatasetsDto,
  ): Promise<CreateRawDatasetsDtoResponse[]> {
    if (data.genbank?.ExInClassifier?.active) {
      const origin = OriginEnum.GENBANK;
      const approach = ApproachEnum.EXINCLASSIFIER;
      if (data.genbank.ExInClassifier.gpt) {
        const modelType = ModelTypeEnum.GPT;
        const maxLength =
          this.configService.getDatasetsLengths().EXINCLASSIFIER.gpt;
        const parentDataset = await this.parentDatasetService.create({
          approach,
          modelType,
          origin,
          name: `${approach}-${modelType}`,
        });

        const exin = await this.FeatureSequenceService.findExIn(maxLength);

        await this.parentRecordService.createMany(
          exin.map((record) => ({
            parentDatasetId: parentDataset.id,
            sequence: record.sequence,
            target: record.type,
            organism: record.organism,
            gene: record.gene,
            flankBefore: record.before,
            flankAfter: record.after,
          })),
        );

        await this.parentDatasetService.update(parentDataset.id, {
          recordCount: exin.length,
        });
      }
      if (data.genbank.ExInClassifier.bert) {
        const modelType = ModelTypeEnum.BERT;
        const maxLength =
          this.configService.getDatasetsLengths().EXINCLASSIFIER.bert;
        const parentDataset = await this.parentDatasetService.create({
          approach,
          modelType,
          origin,
          name: `${approach}-${modelType}`,
        });

        const exin = await this.FeatureSequenceService.findExIn(maxLength);

        await this.parentRecordService.createMany(
          exin.map((record) => ({
            parentDatasetId: parentDataset.id,
            sequence: record.sequence,
            target: record.type,
            organism: record.organism,
            gene: record.gene,
            flankBefore: record.before,
            flankAfter: record.after,
          })),
        );

        await this.parentDatasetService.update(parentDataset.id, {
          recordCount: exin.length,
        });
      }
      if (data.genbank.ExInClassifier.dnabert) {
        const modelType = ModelTypeEnum.DNABERT;
        const maxLength =
          this.configService.getDatasetsLengths().EXINCLASSIFIER.dnabert;
        const parentDataset = await this.parentDatasetService.create({
          approach,
          modelType,
          origin,
          name: `${approach}-${modelType}`,
        });

        const exin = await this.FeatureSequenceService.findExIn(maxLength);

        await this.parentRecordService.createMany(
          exin.map((record) => ({
            parentDatasetId: parentDataset.id,
            sequence: record.sequence,
            target: record.type,
          })),
        );

        await this.parentDatasetService.update(parentDataset.id, {
          recordCount: exin.length,
        });
      }
    }
    if (data.genbank?.TripletClassifier?.active) {
      if (data.genbank.TripletClassifier.bert) {
      }
      if (data.genbank.TripletClassifier.dnabert) {
      }
    }
    if (data.genbank?.DNATranslator?.active) {
      if (data.genbank.DNATranslator.gpt) {
      }
    }
    return [];
  }
}
