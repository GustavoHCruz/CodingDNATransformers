import { Body, Controller, Delete, Get, Param, Patch } from '@nestjs/common';
import { UpdateModelHistoryDto } from './dto/updateModelHistory.dto';
import { ModelHistoryService } from './model-history.service';

@Controller('model-history')
export class ModelHistoryController {
  constructor(private readonly modelHistoryyService: ModelHistoryService) {}

  @Get()
  findAll() {
    return this.modelHistoryyService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.modelHistoryyService.findOne(+id);
  }

  @Patch(':id')
  update(@Param('id') id: string, @Body() data: UpdateModelHistoryDto) {
    return this.modelHistoryyService.update(+id, data);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.modelHistoryyService.remove(+id);
  }
}
