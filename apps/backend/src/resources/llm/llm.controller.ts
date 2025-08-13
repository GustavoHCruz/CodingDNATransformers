import { Controller, Post } from '@nestjs/common';
import { CreateModelDto } from './dto/createModel.dto';
import { LlmService } from './llm.service';

@Controller('llm')
export class LlmController {
  constructor(private readonly llmService: LlmService) {}

  @Post()
  async createModel(data: CreateModelDto) {
    //await this.llmService.generateCsvFromChildDataset('ExIn-GPT', 4);
    //await this.llmService.generateCsvFromChildDataset('ExIn-BERT', 5);
    //await this.llmService.generateCsvFromChildDataset('ExIn-DNABERT', 6);
    //await this.llmService.generateCsvFromChildDataset('Triplet-BERT', 10);
    await this.llmService.generateCsvFromChildDataset('Triplet-DNABERT', 11);
    return true;
  }
}
