import { Module } from '@nestjs/common';
import { GenerationBatchController } from './generation-batch.controller';
import { GenerationBatchRepository } from './generation-batch.repository';
import { GenerationBatchService } from './generation-batch.service';

@Module({
  controllers: [GenerationBatchController],
  providers: [GenerationBatchService, GenerationBatchRepository],
  exports: [GenerationBatchService],
})
export class GenerationBatchModule {}
