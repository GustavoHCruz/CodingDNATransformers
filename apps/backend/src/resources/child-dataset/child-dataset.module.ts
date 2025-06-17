import { Module } from '@nestjs/common';
import { ChildDatasetController } from './child-dataset.controller';
import { ChildDatasetRepository } from './child-dataset.repository';
import { ChildDatasetService } from './child-dataset.service';

@Module({
  controllers: [ChildDatasetController],
  providers: [ChildDatasetRepository, ChildDatasetService],
  exports: [ChildDatasetService],
})
export class ChildDatasetModule {}
