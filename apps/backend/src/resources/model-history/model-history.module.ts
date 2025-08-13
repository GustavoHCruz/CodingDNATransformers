import { Module } from '@nestjs/common';
import { ModelHistoryController } from './model-history.controller';
import { ModelHistoryRepository } from './model-history.repository';
import { ModelHistoryService } from './model-history.service';

@Module({
  controllers: [ModelHistoryController],
  providers: [ModelHistoryRepository, ModelHistoryService],
  exports: [ModelHistoryService],
})
export class ModelHistoryModule {}
