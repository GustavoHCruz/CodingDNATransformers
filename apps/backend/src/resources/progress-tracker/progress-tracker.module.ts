import { Module } from '@nestjs/common';
import { ProgressTrackerController } from './progress-tracker.controller';
import { ProgressTrackerRepository } from './progress-tracker.repository';
import { ProgressTrackerService } from './progress-tracker.service';

@Module({
  controllers: [ProgressTrackerController],
  providers: [ProgressTrackerRepository, ProgressTrackerService],
})
export class ProgressTrackerModule {}
