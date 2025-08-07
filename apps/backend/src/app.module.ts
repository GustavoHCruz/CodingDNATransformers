import { ConfigModule } from '@config/config.module';
import { ExtractionGrpcClientModule } from '@grpc/data-extraction/extraction.grpc-client.module';
import { Module } from '@nestjs/common';
import { PrismaModule } from '@prisma/prisma.module';
import { RedisModule } from '@redis/redis.module';
import { ChildDatasetModule } from '@resources/child-dataset/child-dataset.module';
import { ChildRecordModule } from '@resources/child-record/child-record.module';
import { DataExtractionModule } from '@resources/data-extraction/data-extraction.module';
import { DatasetGenerationModule } from '@resources/dataset-generation/dataset-generation.module';
import { GenerationBatchModule } from '@resources/generation-batch/generation-batch.module';
import { ModelHistoryModule } from '@resources/model-history/model-history.module';
import { ParentDatasetModule } from '@resources/parent-dataset/parent-dataset.module';
import { ParentRecordModule } from '@resources/parent-record/parent-record.module';
import { ProgressTrackerModule } from '@resources/progress-tracker/progress-tracker.module';
import { RawFileInfoModule } from '@resources/raw-file-info/raw-file-info.module';
import { TrainHistoryModule } from '@resources/train-history/train-history.module';
import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [
    ConfigModule,
    ChildDatasetModule,
    ChildRecordModule,
    DataExtractionModule,
    DatasetGenerationModule,
    ExtractionGrpcClientModule,
    GenerationBatchModule,
    ModelHistoryModule,
    ParentDatasetModule,
    ParentRecordModule,
    PrismaModule,
    ProgressTrackerModule,
    TrainHistoryModule,
    RawFileInfoModule,
    RedisModule,
    //LlmModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
