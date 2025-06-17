import { ApproachEnum } from '@prisma/client';
import { IsEnum, IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateChildDatasetDto {
  @IsString()
  name: string;

  @IsEnum(ApproachEnum)
  approach: ApproachEnum;

  @IsNumber()
  batchId: number;

  @IsOptional()
  @IsNumber()
  recordCount?: number;
}
