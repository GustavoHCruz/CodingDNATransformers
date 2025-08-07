import { Module } from '@nestjs/common';
import { FeatureSequenceRepository } from './feature-sequence.repository';
import { FeatureSequenceService } from './feature-sequence.service';

@Module({
  providers: [FeatureSequenceRepository, FeatureSequenceService],
  exports: [FeatureSequenceService],
})
export class FeatureSequenceModule {}
