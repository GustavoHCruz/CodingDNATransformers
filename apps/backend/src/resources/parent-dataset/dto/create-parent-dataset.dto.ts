import { ApproachEnum, ModelTypeEnum, OriginEnum } from '@prisma/client';
import { IsEnum, IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateParentDatasetDto {
  @IsString()
  name: string;

  @IsEnum(ApproachEnum)
  approach: ApproachEnum;

  @IsEnum(ModelTypeEnum)
  modelType: ModelTypeEnum;

  @IsEnum(OriginEnum)
  origin: OriginEnum;

  @IsOptional()
  @IsNumber()
  recordCount?: number;
}
