import { Controller, Post } from '@nestjs/common';
import { CreateModelDto } from './dto/createModel.dto';
import { LlmService } from './llm.service';

@Controller('llm')
export class LlmController {
  constructor(private readonly llmService: LlmService) {}

  @Post()
  createModel(data: CreateModelDto) {
    return this.llmService;
  }
}
