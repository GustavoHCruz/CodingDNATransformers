import { ExtractionGrpcClientModule } from '@grpc/extraction.grpc-client.module';
import { Module } from '@nestjs/common';
import { ParentDatasetModule } from '@resources/parent-dataset/parent-dataset.module';
import { ParentRecordModule } from '@resources/parent-record/parent-record.module';
import { ProgressTrackerModule } from '@resources/progress-tracker/progress-tracker.module';
import { RawFileInfoModule } from '@resources/raw-file-info/raw-file-info.module';
import { ConfigModule } from 'config/config.module';
import { DataExtractionController } from './data-extraction.controller';
import { DataExtractionService } from './data-extraction.service';

@Module({
  imports: [
    ConfigModule,
    ParentDatasetModule,
    ParentRecordModule,
    ProgressTrackerModule,
    RawFileInfoModule,
    ExtractionGrpcClientModule,
  ],
  controllers: [DataExtractionController],
  providers: [DataExtractionService],
})
export class DataExtractionModule {}
