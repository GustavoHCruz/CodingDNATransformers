import { Body, Controller, Post } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { DatasetGenerationService } from './dataset-generation.service';
import {
  CreateDatasetGenerationDto,
  CreateDatasetGenerationDtoResponse,
} from './dto/create-dataset-generation.dto';

@Controller('dataset-generation')
export class DatasetGenerationController {
  constructor(
    private readonly datasetGenerationService: DatasetGenerationService,
  ) {}

  @ApiTags('dataset-generation')
  @Post()
  async createDatasetGeneration(
    @Body() data: CreateDatasetGenerationDto,
  ): Promise<CreateDatasetGenerationDtoResponse[]> {
    return await this.datasetGenerationService.generateProcessedDatasets(data);
  }
}
