import { Injectable } from '@nestjs/common';
import { CreateParentDatasetDto } from './dto/create-parent-dataset.dto';
import { UpdateParentDatasetDto } from './dto/update-parent-dataset.dto';
import { ParentDatasetRepository } from './parent-dataset.repository';

@Injectable()
export class ParentDatasetService {
  constructor(private repository: ParentDatasetRepository) {}

  create(createParentDatasetDto: CreateParentDatasetDto) {
    return this.repository.create(createParentDatasetDto);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  update(id: number, updateParentDatasetDto: UpdateParentDatasetDto) {
    return this.repository.update(id, updateParentDatasetDto);
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
