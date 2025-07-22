import { IsInt } from 'class-validator';

export class TrainModelDto {
  @IsInt()
  childDatasetId: number;
}
