import { Controller, Get, Query } from '@nestjs/common';
import { ApproachEnum } from '@prisma/client';
import { ParentRecordService } from './parent-record.service';

@Controller('parent-record')
export class ParentRecordController {
  constructor(private readonly parentRecordService: ParentRecordService) {}

  @Get('count')
  async count(
    @Query('approach') approach: ApproachEnum,
  ): Promise<{ count: number }> {
    const count = await this.parentRecordService.countByApproach(approach);
    return { count };
  }
}
