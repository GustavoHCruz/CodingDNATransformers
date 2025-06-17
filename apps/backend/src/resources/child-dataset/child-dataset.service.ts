import { Injectable } from '@nestjs/common';
import { ChildDatasetRepository } from './child-dataset.repository';
import { CreateChildDatasetDto } from './dto/create-child-dataset.dto';
import { UpdateChildDatasetDto } from './dto/update-child-dataset.dto';

@Injectable()
export class ChildDatasetService {
  constructor(private repository: ChildDatasetRepository) {}

  create(createChildDatasetDto: CreateChildDatasetDto) {
    return this.repository.create(createChildDatasetDto);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  update(id: number, updateChildDatasetDto: UpdateChildDatasetDto) {
    return this.repository.update(id, updateChildDatasetDto);
  }

  setRecordCount(id: number, counter: number) {
    return this.repository.update(id, {
      recordCount: counter,
    });
  }

  remove(id: number) {
    return this.repository.remove(id);
  }
}
