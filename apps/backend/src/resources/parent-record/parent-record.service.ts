import { Injectable } from '@nestjs/common';
import { CreateParentRecordDto } from './dto/create-parent-record.dto';
import { ParentRecordRepository } from './parent-record.repository';

@Injectable()
export class ParentRecordService {
  constructor(private repository: ParentRecordRepository) {}

  create(createParentRecordDto: CreateParentRecordDto) {
    return this.repository.create(createParentRecordDto);
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
}
