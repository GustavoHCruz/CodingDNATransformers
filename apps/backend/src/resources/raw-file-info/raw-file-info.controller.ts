import { Controller, Get, Param } from '@nestjs/common';
import { RawFileInfoService } from './raw-file-info.service';

@Controller('raw-file-info')
export class RawFileInfoController {
  constructor(private readonly rawFileInfoService: RawFileInfoService) {}

  @Get()
  findAll() {
    return this.rawFileInfoService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.rawFileInfoService.findOne(+id);
  }
}
