import { Controller } from '@nestjs/common';
import { ChildRecordService } from './child-record.service';

@Controller('child-record')
export class ChildRecordController {
  constructor(private readonly childRecordService: ChildRecordService) {}
}
