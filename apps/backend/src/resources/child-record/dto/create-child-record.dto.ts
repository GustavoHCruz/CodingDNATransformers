import { IsNumber } from 'class-validator';

export class CreateChildRecordDto {
  @IsNumber()
  childDatasetId: number;

  @IsNumber()
  parentRecordId: number;
}
