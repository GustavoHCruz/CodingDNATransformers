import { ExtractionGrpcClientModule } from '@grpc/data-extraction/extraction.grpc-client.module';
import { Module } from '@nestjs/common';
import { DnaSequenceModule } from '@resources/dna-sequence/dna-sequence.module';
import { FeatureSequenceModule } from '@resources/feature-sequence/dna-sequence.module';
import { ProgressTrackerModule } from '@resources/progress-tracker/progress-tracker.module';
import { RawFileInfoModule } from '@resources/raw-file-info/raw-file-info.module';
import { ConfigModule } from 'config/config.module';
import { DataExtractionController } from './data-extraction.controller';
import { DataExtractionService } from './data-extraction.service';

@Module({
  imports: [
    ConfigModule,
    DnaSequenceModule,
    FeatureSequenceModule,
    ProgressTrackerModule,
    RawFileInfoModule,
    ExtractionGrpcClientModule,
  ],
  controllers: [DataExtractionController],
  providers: [DataExtractionService],
})
export class DataExtractionModule {}
