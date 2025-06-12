import { Controller, Delete, Get, Param } from '@nestjs/common';
import { ProgressTrackerService } from './progress-tracker.service';

@Controller('progress-tracker')
export class ProgressTrackerController {
  constructor(
    private readonly progressTrackerService: ProgressTrackerService,
  ) {}

  @Get()
  findAll() {
    return this.progressTrackerService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.progressTrackerService.findOne(+id);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.progressTrackerService.remove(+id);
  }
}
