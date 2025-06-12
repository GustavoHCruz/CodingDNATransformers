import { Injectable } from '@nestjs/common';
import { ApproachEnum, OriginEnum } from '@prisma/client';
import { ParentDatasetService } from '@resources/parent-dataset/parent-dataset.service';
import { ExtractionGrpcClientService } from 'gprc/extraction.grpc-client.service';
import { firstValueFrom } from 'rxjs';
import {
  CreateDataExtractionDto,
  CreateDataExtractionResponseDto,
} from './dto/data-extraction.dto';

type ExtractionTask = {
  function: any,
  annotationsPath: string,
  approach: ApproachEnum,
  origin: OriginEnum
}

@Injectable()
export class DataExtractionService {
  constructor(
    private readonly extractionClient: ExtractionGrpcClientService,
    private parentDataset: ParentDatasetService,
    private task,
  ) {}

  async extractionTask({
    function,
    annotationsPath,
    approach,
    origin
  }: ExtractionTask): number {

  }

  async extract(
    data: CreateDataExtractionDto,
  ): Promise<CreateDataExtractionResponseDto> {
    const response = new CreateDataExtractionResponseDto();

    const genbank = data.genbank;
    const gencode = data.gencode;
    if (genbank) {
      if (genbank.ExInClassifier) {
        origin = OriginEnum.GENBANK;

        response.genbank.ExInClassifier = 1
      }
    }

    if (data.genbank?.ExInClassifier) {
      const response = firstValueFrom(
        this.extractionClient.call('ExInClassifierGenbank', {
          parentId: 1,
          sequenceMaxLength: 512,
        }),
      ).then((value) => {});
    }

    return response;
  }
}
