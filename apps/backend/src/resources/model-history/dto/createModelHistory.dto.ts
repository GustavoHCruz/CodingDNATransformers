import { ApproachEnum } from '@prisma/client';
import { IsEnum, IsString } from 'class-validator';

export class CreateModelHistoryDto {
  @IsString()
  modelAlias: string;

  @IsString()
  checkpointName: string;

  @IsEnum(ApproachEnum)
  approach: ApproachEnum;

  @IsString()
  path: string;
}
