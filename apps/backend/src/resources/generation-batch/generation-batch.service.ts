import { Injectable } from '@nestjs/common';
import { CreateGenerationBatchDto } from './dto/create-generation-batch.dto';
import { UpdateGenerationBatchDto } from './dto/update-generation-batch.dto';
import { GenerationBatchRepository } from './generation-batch.repository';

@Injectable()
export class GenerationBatchService {
  constructor(private repository: GenerationBatchRepository) {}

  create(data: CreateGenerationBatchDto) {
    return this.repository.create(data);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  update(id: number, data: UpdateGenerationBatchDto) {
    return this.repository.update(id, data);
  }

  remove(id: number) {
    return this.repository.remove(id);
  }
}
