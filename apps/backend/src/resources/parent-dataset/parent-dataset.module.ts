import { Module } from '@nestjs/common';
import { ParentDatasetController } from './parent-dataset.controller';
import { ParentDatasetRepository } from './parent-dataset.repository';
import { ParentDatasetService } from './parent-dataset.service';

@Module({
  controllers: [ParentDatasetController],
  providers: [ParentDatasetRepository, ParentDatasetService],
  exports: [ParentDatasetService],
})
export class ParentDatasetModule {}
