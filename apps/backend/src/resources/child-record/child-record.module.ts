import { Module } from '@nestjs/common';
import { ChildRecordController } from './child-record.controller';
import { ChildRecordRepository } from './child-record.repository';
import { ChildRecordService } from './child-record.service';

@Module({
  controllers: [ChildRecordController],
  providers: [ChildRecordRepository, ChildRecordService],
  exports: [ChildRecordService],
})
export class ChildRecordModule {}
