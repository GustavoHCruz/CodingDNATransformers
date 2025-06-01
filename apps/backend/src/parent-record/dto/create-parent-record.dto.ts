import { IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateParentRecordDto {
  @IsString()
  sequence: string;

  @IsString()
  target: string;

  @IsOptional()
  @IsString()
  flankBefore?: string;

  @IsOptional()
  @IsString()
  flankAfter?: string;

  @IsOptional()
  @IsString()
  organism?: string;

  @IsOptional()
  @IsString()
  gene?: string;

  @IsNumber()
  parentDatasetId: number;
}
