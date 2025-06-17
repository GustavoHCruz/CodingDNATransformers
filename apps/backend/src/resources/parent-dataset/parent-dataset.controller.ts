import { Body, Controller, Delete, Get, Param, Patch } from '@nestjs/common';
import { UpdateParentDatasetDto } from './dto/update-parent-dataset.dto';
import { ParentDatasetService } from './parent-dataset.service';

@Controller('parent-dataset')
export class ParentDatasetController {
  constructor(private readonly parentDatasetService: ParentDatasetService) {}

  @Get()
  findAll() {
    return this.parentDatasetService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.parentDatasetService.findOne(+id);
  }

  @Patch(':id')
  update(
    @Param('id') id: string,
    @Body() updateParentDatasetDto: UpdateParentDatasetDto,
  ) {
    return this.parentDatasetService.update(+id, updateParentDatasetDto);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.parentDatasetService.remove(+id);
  }
}
