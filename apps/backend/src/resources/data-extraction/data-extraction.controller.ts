import { Controller, Post } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { DataExtractionService } from './data-extraction.service';
import { DataExtractionReturn } from './dto/approachs.dto';

@Controller('data-extraction')
export class DataExtractionController {
  constructor(private readonly dataExtractionService: DataExtractionService) {}

  @ApiTags('data-extraction')
  @Post()
  async dataExtraction(): Promise<DataExtractionReturn> {
    return await this.dataExtractionService.extract();
  }
}
