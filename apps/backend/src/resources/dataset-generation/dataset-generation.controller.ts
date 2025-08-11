import { Body, Controller, Post } from '@nestjs/common';
import { DatasetGenerationService } from './dataset-generation.service';
import {
  CreateProcessedDatasetsDto,
  CreateProcessedDatasetsDtoResponse,
} from './dto/create-processed-datasets.dto';
import {
  CreateRawDatasetsDto,
  CreateRawDatasetsDtoResponse,
} from './dto/create-raw-datasets.dto';

@Controller('dataset-generation')
export class DatasetGenerationController {
  constructor(
    private readonly datasetGenerationService: DatasetGenerationService,
  ) {}

  @Post('/raw-datasets')
  async generateRawDatasets(
    @Body() data: CreateRawDatasetsDto,
  ): Promise<CreateRawDatasetsDtoResponse> {
    return this.datasetGenerationService.generateRawDatasets(data);
  }

  @Post('/processed-datasets')
  async generateProcessedDatasets(
    @Body() data: CreateProcessedDatasetsDto,
  ): Promise<CreateProcessedDatasetsDtoResponse[]> {
    return await this.datasetGenerationService.generateProcessedDatasets(data);
  }
}
