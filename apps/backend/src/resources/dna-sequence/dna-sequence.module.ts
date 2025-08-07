import { Module } from '@nestjs/common';
import { DnaSequenceRepository } from './dna-sequence.repository';
import { DnaSequenceService } from './dna-sequence.service';

@Module({
  providers: [DnaSequenceRepository, DnaSequenceService],
  exports: [DnaSequenceService],
})
export class DnaSequenceModule {}
