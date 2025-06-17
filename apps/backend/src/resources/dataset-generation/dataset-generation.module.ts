import { ConfigModule } from '@config/config.module';
import { Module } from '@nestjs/common';
import { ChildDatasetModule } from '@resources/child-dataset/child-dataset.module';
import { ChildRecordModule } from '@resources/child-record/child-record.module';
import { GenerationBatchModule } from '@resources/generation-batch/generation-batch.module';
import { ParentRecordModule } from '@resources/parent-record/parent-record.module';
import { ProgressTrackerModule } from '@resources/progress-tracker/progress-tracker.module';
import { DatasetGenerationController } from './dataset-generation.controller';
import { DatasetGenerationService } from './dataset-generation.service';

@Module({
  imports: [
    ProgressTrackerModule,
    ChildDatasetModule,
    ChildRecordModule,
    GenerationBatchModule,
    ConfigModule,
    ParentRecordModule,
  ],
  controllers: [DatasetGenerationController],
  providers: [DatasetGenerationService, DatasetGenerationController],
  exports: [DatasetGenerationService],
})
export class DatasetGenerationModule {}
