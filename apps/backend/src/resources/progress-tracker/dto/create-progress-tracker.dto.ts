import { ProgressStatusEnum, ProgressTypeEnum } from '@prisma/client';
import { IsEnum, IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateProgressTrackerDto {
  @IsOptional()
  @IsString()
  taskName?: string;

  @IsOptional()
  @IsNumber()
  progress?: number;

  @IsOptional()
  @IsEnum(ProgressTypeEnum)
  progressType?: ProgressTypeEnum;

  @IsOptional()
  @IsEnum(ProgressStatusEnum)
  status?: ProgressStatusEnum;

  @IsOptional()
  @IsString()
  info?: string;
}
