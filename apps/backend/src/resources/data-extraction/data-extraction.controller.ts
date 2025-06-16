import { Body, Controller, Post } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { DataExtractionService } from './data-extraction.service';
import {
  CreateDataExtractionDto,
  CreateDataExtractionResponseDto,
} from './dto/data-extraction.dto';

@Controller('data-extraction')
export class DataExtractionController {
  constructor(private readonly dataExtractionService: DataExtractionService) {}

  @ApiTags('data-extraction')
  @Post()
  async dataExtraction(
    @Body() data: CreateDataExtractionDto,
  ): Promise<CreateDataExtractionResponseDto> {
    return await this.dataExtractionService.extract(data);
  }
}
