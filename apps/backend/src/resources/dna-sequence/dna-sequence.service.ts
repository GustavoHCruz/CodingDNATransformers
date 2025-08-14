import { Injectable } from '@nestjs/common';
import { DnaSequenceRepository } from './dna-sequence.repository';
import {
  CreateDnaSequenceDto,
  CreateDnaSequenceWithFeatureBatchDto,
} from './dto/create-dna-sequence.dto';

@Injectable()
export class DnaSequenceService {
  constructor(private repository: DnaSequenceRepository) {}
  create(data: CreateDnaSequenceDto) {
    return this.repository.create(data);
  }

  createWithFeatureBatch(batch: CreateDnaSequenceWithFeatureBatchDto[]) {
    return this.repository.createWithFeatureBatch(batch);
  }

  createMany(data: CreateDnaSequenceDto[]) {
    return this.repository.createMany(data);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  remove(id: number) {
    return this.repository.remove(id);
  }

  findTriplet(maxLength: number, limit: number, lastId: number | null) {
    return this.repository.findTriplet(maxLength, limit, lastId);
  }
}
