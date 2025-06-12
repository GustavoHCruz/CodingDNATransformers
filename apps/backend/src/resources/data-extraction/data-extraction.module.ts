import { Module } from '@nestjs/common';
import { DataExtractionController } from './data-extraction.controller';
import { DataExtractionService } from './data-extraction.service';

@Module({
  controllers: [DataExtractionController],
  providers: [DataExtractionService],
})
export class ProgressTrackerModule {}
