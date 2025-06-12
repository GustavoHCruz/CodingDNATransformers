import { Controller, Post } from '@nestjs/common';
import { DataExtractionService } from './data-extraction.service';
import {
  CreateDataExtractionDto,
  CreateDataExtractionResponseDto,
} from './dto/data-extraction.dto';

@Controller('data-extraction')
export class DataExtractionController {
  constructor(private readonly dataExtractionService: DataExtractionService) {}

  @Post()
  async dataExtraction(
    data: CreateDataExtractionDto,
  ): Promise<CreateDataExtractionResponseDto> {
    return await this.dataExtractionService.extract(data);
  }
}
