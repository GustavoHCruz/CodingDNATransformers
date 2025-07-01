import { IsString } from 'class-validator';

export class UpdateModelHistoryDto {
  @IsString()
  modelAlias: string;

  @IsString()
  checkpointName: string;
}
