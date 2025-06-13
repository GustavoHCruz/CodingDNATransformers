import { Injectable } from '@nestjs/common';
import { CreateParentRecordDto } from './dto/create-parent-record.dto';
import { ParentRecordRepository } from './parent-record.repository';

@Injectable()
export class ParentRecordService {
  constructor(private repository: ParentRecordRepository) {}
  private seenByTask = new Map<number, Set<string>>();

  create(data: CreateParentRecordDto) {
    return this.repository.create(data);
  }

  createMany(data: CreateParentRecordDto[]) {
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

  removeDuplicated(data: CreateParentRecordDto[], taskId: number) {
    if (!this.seenByTask.has(taskId)) {
      this.seenByTask.set(taskId, new Set());
    }

    const seen = this.seenByTask.get(taskId)!;
    const unique: CreateParentRecordDto[] = [];

    const keyFields = [
      'sequence',
      'target',
      'flank_before',
      'flank_after',
      'organism',
      'gene',
    ];

    for (const instance of data) {
      const key = keyFields.map((field) => instance[field] || '').join('|');

      if (!seen.has(key)) {
        seen.add(key);
        unique.push(instance);
      }
    }

    return unique;
  }

  clearTaskState(taskId: number) {
    this.seenByTask.delete(taskId);
  }
}
