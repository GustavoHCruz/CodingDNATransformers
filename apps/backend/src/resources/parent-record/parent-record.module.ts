import { Module } from '@nestjs/common';
import { ParentRecordController } from './parent-record.controller';
import { ParentRecordRepository } from './parent-record.repository';
import { ParentRecordService } from './parent-record.service';

@Module({
  controllers: [ParentRecordController],
  providers: [ParentRecordRepository, ParentRecordService],
  exports: [ParentRecordService],
})
export class ParentRecordModule {}
