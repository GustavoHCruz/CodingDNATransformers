import { Body, Controller, Post } from '@nestjs/common';
import { CreateCsvFromParentDataset } from './dto/createModel.dto';
import { LlmService } from './llm.service';

@Controller('llm')
export class LlmController {
  constructor(private readonly llmService: LlmService) {}

  @Post('/generate-csv')
  async generateCsv(@Body() data: CreateCsvFromParentDataset) {
    return this.llmService.generateCsv(data);
  }
}
