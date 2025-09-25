import { ConfigService } from '@config/config.service';
import { ForbiddenException, Injectable } from '@nestjs/common';
import {
  ApproachEnum,
  FeatureEnum,
  ModelTypeEnum,
  OriginEnum,
  ProgressTypeEnum,
} from '@prisma/client';
import { ChildDatasetService } from '@resources/child-dataset/child-dataset.service';
import { ChildRecordService } from '@resources/child-record/child-record.service';
import { DnaSequenceService } from '@resources/dna-sequence/dna-sequence.service';
import { FeatureSequenceService } from '@resources/feature-sequence/feature-sequence.service';
import { GenerationBatchService } from '@resources/generation-batch/generation-batch.service';
import { ParentDatasetService } from '@resources/parent-dataset/parent-dataset.service';
import { CreateParentRecordDto } from '@resources/parent-record/dto/create-parent-record.dto';
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
    private readonly DNASequenceService: DnaSequenceService,
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

  async generateRawDatasetsExInFn(
    approach: ApproachEnum,
    modelType: ModelTypeEnum,
    origin: OriginEnum,
    maxLength: number,
    batchSize: number,
    taskId: number,
  ) {
    const parentDatasetId = (
      await this.parentDatasetService.create({
        approach,
        modelType,
        origin,
        name: `${approach}-${modelType}`,
      })
    ).id;

    let lastId: number | null = null;
    let total = 0;
    while (true) {
      const batch = await this.FeatureSequenceService.findExIn(
        maxLength,
        batchSize,
        lastId,
      );

      if (!batch.length) {
        break;
      }

      const result = await this.parentRecordService.createMany(
        batch.map((record) => ({
          parentDatasetId,
          sequence: record.sequence,
          target: record.type,
          organism: record.dnaSequence?.organism || '',
          gene: record.gene || '',
          flankBefore: record.before || '',
          flankAfter: record.after || '',
        })),
      );

      lastId = batch.length > 0 ? batch[batch.length - 1].id : null;
      total += result.count;

      await this.progressService.postProgress(taskId, total);
    }
    await this.parentDatasetService.update(parentDatasetId, {
      recordCount: total,
    });
    await this.progressService.finish(taskId);
  }

  async generateRawDatasetsTripletFn(
    approach: ApproachEnum,
    modelType: ModelTypeEnum,
    origin: OriginEnum,
    maxLength: number,
    batchSize: number,
    taskId: number,
  ) {
    const parentDatasetId = (
      await this.parentDatasetService.create({
        approach,
        modelType,
        origin,
        name: `${approach}-${modelType}`,
      })
    ).id;

    let lastId: number | null = null;
    let total = 0;
    while (true) {
      const batch = await this.DNASequenceService.findTriplet(
        maxLength,
        batchSize,
        lastId,
      );

      if (!batch.length) {
        break;
      }

      const result = await this.parentRecordService.createMany(
        batch.map((record) => {
          const splicingSeq = Array(record.sequence.length).fill('U');

          for (const feat of record.features) {
            const start = feat.start;
            const end = feat.end;
            const char = feat.type === FeatureEnum.INTRON ? 'I' : 'E';
            splicingSeq.fill(char, start, end);
          }

          const target = splicingSeq.join('');

          return {
            parentDatasetId,
            sequence: record.sequence,
            target,
            organism: record.organism || '',
          };
        }),
      );

      lastId = batch[batch.length - 1].id;
      total += result.count;

      await this.progressService.postProgress(taskId, total);
    }
    await this.parentDatasetService.update(parentDatasetId, {
      recordCount: total,
    });
    await this.progressService.finish(taskId);
  }

  async generateRawDatasetsDNATranslatorFn(
    approach: ApproachEnum,
    modelType: ModelTypeEnum,
    origin: OriginEnum,
    maxLength: number,
    batchSize: number,
    taskId: number,
  ) {
    const parentDatasetId = (
      await this.parentDatasetService.create({
        approach,
        modelType,
        origin,
        name: `${approach}-${modelType}`,
      })
    ).id;

    const proteinMaxLength = Math.floor(maxLength / 3);

    let lastId: number | null = null;
    let total = 0;
    while (true) {
      const batch = await this.FeatureSequenceService.findWithCDS(
        maxLength,
        proteinMaxLength,
        batchSize,
        lastId,
      );

      if (!batch.length) {
        break;
      }

      const batchToCreate: CreateParentRecordDto[] = [];
      for (const record of batch) {
        const target = record.features.reduce(
          (acc, feature) => (acc += feature.sequence),
          '',
        );
        if (target.length > proteinMaxLength) {
          continue;
        }

        const sequence = record.sequence;
        const organism = record.organism || '';

        batchToCreate.push({
          parentDatasetId,
          sequence: sequence,
          target: target,
          organism: organism,
        });
      }

      const result = await this.parentRecordService.createMany(batchToCreate);

      lastId = batch[batch.length - 1].id;
      total += result.count;

      await this.progressService.postProgress(taskId, total);
    }
    await this.parentDatasetService.update(parentDatasetId, {
      recordCount: total,
    });
    await this.progressService.finish(taskId);
  }

  async generateRawDatasets(
    data: CreateRawDatasetsDto,
  ): Promise<CreateRawDatasetsDtoResponse> {
    const response: CreateRawDatasetsDtoResponse = {
      genbank: {
        ExInClassifier: {},
        TripletClassifier: {},
        DNATranslator: {},
      },
    };

    const batchSize = this.configService.getDatasetGeneration().batch_size;

    const origin = OriginEnum.GENBANK;
    if (data.genbank?.ExInClassifier?.active) {
      const approach = ApproachEnum.EXINCLASSIFIER;
      if (data.genbank.ExInClassifier.gpt) {
        const modelType = ModelTypeEnum.GPT;
        const maxLength =
          this.configService.getDatasetsLengths().EXINCLASSIFIER.gpt;

        const taskId = (
          await this.progressService.create({
            progressType: ProgressTypeEnum.COUNTER,
          })
        ).id;
        this.generateRawDatasetsExInFn(
          approach,
          modelType,
          origin,
          maxLength,
          batchSize,
          taskId,
        );
        response.genbank.ExInClassifier!.gpt = taskId;
      }
      if (data.genbank.ExInClassifier.bert) {
        const modelType = ModelTypeEnum.BERT;
        const maxLength =
          this.configService.getDatasetsLengths().EXINCLASSIFIER.bert;

        const taskId = (
          await this.progressService.create({
            progressType: ProgressTypeEnum.COUNTER,
          })
        ).id;
        this.generateRawDatasetsExInFn(
          approach,
          modelType,
          origin,
          maxLength,
          batchSize,
          taskId,
        );
        response.genbank.ExInClassifier!.bert = taskId;
      }
      if (data.genbank.ExInClassifier.dnabert) {
        const modelType = ModelTypeEnum.DNABERT;
        const maxLength =
          this.configService.getDatasetsLengths().EXINCLASSIFIER.dnabert;

        const taskId = (
          await this.progressService.create({
            progressType: ProgressTypeEnum.COUNTER,
          })
        ).id;
        this.generateRawDatasetsExInFn(
          approach,
          modelType,
          origin,
          maxLength,
          batchSize,
          taskId,
        );
        response.genbank.ExInClassifier!.dnabert = taskId;
      }
    }
    if (data.genbank?.TripletClassifier?.active) {
      const approach = ApproachEnum.TRIPLETCLASSIFIER;
      if (data.genbank.TripletClassifier.bert) {
        const modelType = ModelTypeEnum.BERT;
        const maxLength =
          this.configService.getDatasetsLengths().TRIPLETCLASSIFIER.bert;

        const taskId = (
          await this.progressService.create({
            progressType: ProgressTypeEnum.COUNTER,
          })
        ).id;
        this.generateRawDatasetsTripletFn(
          approach,
          modelType,
          origin,
          maxLength,
          batchSize,
          taskId,
        );
        response.genbank.TripletClassifier!.bert = taskId;
      }
      if (data.genbank.TripletClassifier.dnabert) {
        const modelType = ModelTypeEnum.DNABERT;
        const maxLength =
          this.configService.getDatasetsLengths().TRIPLETCLASSIFIER.dnabert;

        const taskId = (
          await this.progressService.create({
            progressType: ProgressTypeEnum.COUNTER,
          })
        ).id;
        this.generateRawDatasetsTripletFn(
          approach,
          modelType,
          origin,
          maxLength,
          batchSize,
          taskId,
        );
        response.genbank.TripletClassifier!.dnabert = taskId;
      }
    }
    if (data.genbank?.DNATranslator?.active) {
      const approach = ApproachEnum.DNATRANSLATOR;
      if (data.genbank.DNATranslator.gpt) {
        const modelType = ModelTypeEnum.GPT;
        const maxLength =
          this.configService.getDatasetsLengths().DNATRANSLATOR.gpt;

        const taskId = (
          await this.progressService.create({
            progressType: ProgressTypeEnum.COUNTER,
          })
        ).id;
        this.generateRawDatasetsDNATranslatorFn(
          approach,
          modelType,
          origin,
          maxLength,
          batchSize,
          taskId,
        );
        response.genbank.DNATranslator!.gpt = taskId;
      }
    }
    return response;
  }
}
