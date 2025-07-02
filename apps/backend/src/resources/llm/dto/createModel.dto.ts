import { ApproachEnum } from '@prisma/client';
import { IsEnum, IsInt, IsOptional, IsString } from 'class-validator';

export class CreateModelDto {
  @IsString()
  modelAlias: string;

  @IsString()
  checkpointName: string;

  @IsEnum(ApproachEnum)
  approach: ApproachEnum;

  @IsOptional()
  @IsInt()
  parentId: number;
}
