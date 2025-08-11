import { ApproachEnum, ModelTypeEnum } from '@prisma/client';
import { IsEnum, IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateChildDatasetDto {
  @IsString()
  name: string;

  @IsEnum(ApproachEnum)
  approach: ApproachEnum;

  @IsEnum(ModelTypeEnum)
  modelType: ModelTypeEnum;

  @IsNumber()
  batchId: number;

  @IsOptional()
  @IsNumber()
  recordCount?: number;
}
