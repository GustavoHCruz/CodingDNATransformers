import { Module } from '@nestjs/common';
import { TrainHistoryController } from './train-history.controller';
import { TrainHistoryRepository } from './train-history.repository';
import { TrainHistoryService } from './train-history.service';

@Module({
  controllers: [TrainHistoryController],
  providers: [TrainHistoryRepository, TrainHistoryService],
  exports: [TrainHistoryService],
})
export class TrainHistoryModule {}
