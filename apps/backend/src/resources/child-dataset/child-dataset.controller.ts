import { Body, Controller, Delete, Get, Param, Patch } from '@nestjs/common';
import { ChildDatasetService } from './child-dataset.service';
import { UpdateChildDatasetDto } from './dto/update-child-dataset.dto';

@Controller('child-dataset')
export class ChildDatasetController {
  constructor(private readonly childDatasetService: ChildDatasetService) {}

  @Get()
  findAll() {
    return this.childDatasetService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.childDatasetService.findOne(+id);
  }

  @Patch(':id')
  update(
    @Param('id') id: string,
    @Body() updateChildDatasetDto: UpdateChildDatasetDto,
  ) {
    return this.childDatasetService.update(+id, updateChildDatasetDto);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.childDatasetService.remove(+id);
  }
}
