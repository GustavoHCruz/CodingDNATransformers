import { Controller } from '@nestjs/common';
import { ParentRecordService } from './parent-record.service';

@Controller('parent-record')
export class ParentRecordController {
  constructor(private readonly parentRecordService: ParentRecordService) {}
}
