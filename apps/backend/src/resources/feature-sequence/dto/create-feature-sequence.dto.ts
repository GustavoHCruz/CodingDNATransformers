import { FeatureEnum } from '@prisma/client';
import { IsEnum, IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateFeatureSequenceDto {
  @IsString()
  sequence: string;

  @IsNumber()
  length: number;

  @IsEnum(FeatureEnum)
  type: FeatureEnum;

  @IsNumber()
  start: number;

  @IsNumber()
  end: number;

  @IsOptional()
  @IsString()
  gene?: string;

  @IsOptional()
  @IsNumber()
  strand?: number;

  @IsNumber()
  dnaSequenceId: number;
}
