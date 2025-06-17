import { Body, Controller, Get, Param, Patch } from '@nestjs/common';
import { UpdateGenerationBatchDto } from './dto/update-generation-batch.dto';
import { GenerationBatchService } from './generation-batch.service';

@Controller('generation-batch')
export class GenerationBatchController {
  constructor(
    private readonly generationBatchService: GenerationBatchService,
  ) {}

  @Get()
  findAll() {
    return this.generationBatchService.findAll();
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.generationBatchService.findOne(+id);
  }

  @Patch(':id')
  update(
    @Param('id') id: string,
    @Body() updateGenerationBatchDto: UpdateGenerationBatchDto,
  ) {
    return this.generationBatchService.update(+id, updateGenerationBatchDto);
  }
}
