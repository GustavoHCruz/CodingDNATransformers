import { Injectable } from '@nestjs/common';
import { ApproachEnum, Prisma } from '@prisma/client';
import { RawFileInfoRepository } from './raw-file-info.repository';

@Injectable()
export class RawFileInfoService {
  constructor(private repository: RawFileInfoRepository) {}

  create(data: Prisma.RawFileInfoCreateInput) {
    return this.repository.create(data);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  findByFileNameAndApproach(fileName: string, approach: ApproachEnum) {
    return this.repository.findOneByFileNameAndApproach(fileName, approach);
  }

  updateRecordCounting(id: number, recordCounting: number) {
    return this.repository.update(id, {
      totalRecords: recordCounting,
    });
  }
}
