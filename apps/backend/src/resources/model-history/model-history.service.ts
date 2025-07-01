import { Injectable } from '@nestjs/common';
import { CreateModelHistoryDto } from './dto/createModelHistory.dto';
import { UpdateModelHistoryDto } from './dto/updateModelHistory.dto';
import { ModelHistoryRepository } from './model-history.repository';

@Injectable()
export class ModelHistoryService {
  constructor(private repository: ModelHistoryRepository) {}

  create(data: CreateModelHistoryDto) {
    return this.repository.create(data);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  update(id: number, data: UpdateModelHistoryDto) {
    return this.repository.update(id, data);
  }

  remove(id: number) {
    return this.repository.remove(id);
  }
}
