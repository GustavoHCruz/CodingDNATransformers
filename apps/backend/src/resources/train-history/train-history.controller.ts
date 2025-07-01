import { Controller, Get, Param } from '@nestjs/common';
import { TrainHistoryService } from './train-history.service';

@Controller('train-history')
export class TrainHistoryController {
  constructor(private readonly trainHistoryService: TrainHistoryService) {}

  @Get()
  findAll() {
    return this.trainHistoryService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.trainHistoryService.findOne(+id);
  }
}
