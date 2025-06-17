import { ConfigService } from '@config/config.service';
import { Injectable } from '@nestjs/common';
import { ApproachEnum, ProgressTypeEnum } from '@prisma/client';
import { ChildDatasetService } from '@resources/child-dataset/child-dataset.service';
import { ChildRecordService } from '@resources/child-record/child-record.service';
import { GenerationBatchService } from '@resources/generation-batch/generation-batch.service';
import { ParentRecordService } from '@resources/parent-record/parent-record.service';
import { ProgressTrackerService } from '@resources/progress-tracker/progress-tracker.service';
import {
  CreateDatasetGenerationDto,
  CreateDatasetGenerationDtoResponse,
} from './dto/create-dataset-generation.dto';

@Injectable()
export class DatasetGenerationService {
  constructor(
    private readonly progressService: ProgressTrackerService,
    private readonly childDatasetService: ChildDatasetService,
    private readonly childRecordService: ChildRecordService,
    private readonly generationBatchService: GenerationBatchService,
    private readonly configService: ConfigService,
    private readonly parentRecordService: ParentRecordService,
  ) {}

  private async backgroundChildDatasetCreation(
    batchGenerationId: number,
    approach: ApproachEnum,
    childDatasetName: string,
    parentRecordIds: number[],
  ): Promise<number> {
    const totalRecords = parentRecordIds.length;

    const newChildDataset = await this.childDatasetService.create({
      name: childDatasetName,
      approach,
      batchId: batchGenerationId,
      recordCount: totalRecords,
    });

    const task = await this.progressService.create({
      progressType: ProgressTypeEnum.PERCENTAGE,
      taskName: `batch:${batchGenerationId} child:${newChildDataset.id}`,
    });

    const taskId = task.id;
    const childDatasetId = newChildDataset.id;

    const batchSize =
      this.configService.getDatasetGeneration().batch_size || 100;

    (async () => {
      let counter = 0;

      for (let i = 0; i < parentRecordIds.length; i += batchSize) {
        const batch = parentRecordIds.slice(i, i + batchSize);

        const data = batch.map((parentId) => ({
          childDatasetId: childDatasetId,
          batchId: batchGenerationId,
          parentRecordId: parentId,
        }));

        await this.childRecordService.createMany(data);

        counter += batch.length;
        await this.progressService.postProgress(taskId, totalRecords, counter);
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
    data: CreateDatasetGenerationDto,
  ): Promise<CreateDatasetGenerationDtoResponse[]> {
    const responses: CreateDatasetGenerationDtoResponse[] = [];

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

      const taskId = await this.backgroundChildDatasetCreation(
        generationBatchId,
        data.approach,
        child.name,
        parentIdsToUse,
      );

      responses.push({
        name: child.name,
        taskId,
      });

      recordsSplitStart = recordsSplitEnd;
    }

    return responses;
  }
}
