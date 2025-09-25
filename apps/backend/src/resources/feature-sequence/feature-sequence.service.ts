import { Injectable } from '@nestjs/common';
import { CreateFeatureSequenceDto } from './dto/create-feature-sequence.dto';
import { FeatureSequenceRepository } from './feature-sequence.repository';

@Injectable()
export class FeatureSequenceService {
  constructor(private repository: FeatureSequenceRepository) {}
  create(data: CreateFeatureSequenceDto) {
    return this.repository.create(data);
  }

  createMany(data: CreateFeatureSequenceDto[]) {
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

  findExIn(maxLength: number, limit: number, lastId: number | null) {
    return this.repository.findExIn(maxLength, limit, lastId);
  }

  async findWithCDS(
    limit: number,
    dnaMaxLength: number,
    proteinMaxLength: number,
    lastId: number | null,
  ) {
    return this.repository.findWithCDS(
      dnaMaxLength,
      proteinMaxLength,
      limit,
      lastId,
    );
  }
}
