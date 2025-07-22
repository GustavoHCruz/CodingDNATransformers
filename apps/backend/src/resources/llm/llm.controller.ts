import { Controller, Post } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { CreateModelDto } from './dto/createModel.dto';
import { LlmService } from './llm.service';

@Controller('llm')
export class LlmController {
  constructor(private readonly llmService: LlmService) {}

  @Post()
  createModel(data: CreateModelDto) {
    this.llmService.generateCsvFromChildDataset(uuidv4(), 4);
    this.llmService.generateCsvFromChildDataset(uuidv4(), 5);
    return true;
  }
}
