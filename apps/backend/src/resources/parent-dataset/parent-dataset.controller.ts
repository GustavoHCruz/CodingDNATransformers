import { Controller, Get, Post, Body, Patch, Param, Delete } from '@nestjs/common';
import { ParentDatasetService } from './parent-dataset.service';
import { CreateParentDatasetDto } from './dto/create-parent-dataset.dto';
import { UpdateParentDatasetDto } from './dto/update-parent-dataset.dto';

@Controller('parent-dataset')
export class ParentDatasetController {
  constructor(private readonly parentDatasetService: ParentDatasetService) {}

  @Post()
  create(@Body() createParentDatasetDto: CreateParentDatasetDto) {
    return this.parentDatasetService.create(createParentDatasetDto);
  }

  @Get()
  findAll() {
    return this.parentDatasetService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.parentDatasetService.findOne(+id);
  }

  @Patch(':id')
  update(@Param('id') id: string, @Body() updateParentDatasetDto: UpdateParentDatasetDto) {
    return this.parentDatasetService.update(+id, updateParentDatasetDto);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.parentDatasetService.remove(+id);
  }
}
