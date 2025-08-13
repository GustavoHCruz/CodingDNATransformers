import { Injectable } from '@nestjs/common';
import { ChildRecordRepository } from './child-record.repository';
import { CreateChildRecordDto } from './dto/create-child-record.dto';

@Injectable()
export class ChildRecordService {
  constructor(private repository: ChildRecordRepository) {}

  create(data: CreateChildRecordDto) {
    return this.repository.create(data);
  }

  createMany(data: CreateChildRecordDto[]) {
    return this.repository.createMany(data);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  streamFindAllByChildDatasetId(childDatasetId: number) {
    return this.repository.streamFindAllByChildDatasetId(childDatasetId);
  }

  remove(id: number) {
    return this.repository.remove(id);
  }
}
