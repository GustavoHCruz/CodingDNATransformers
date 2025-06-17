import { ApproachEnum } from '@prisma/client';
import { Type } from 'class-transformer';
import {
  IsEnum,
  IsInt,
  IsNotEmpty,
  IsOptional,
  IsString,
  Min,
  ValidateNested,
} from 'class-validator';

export class ChildDatasetSettingsDto {
  @IsString()
  @IsNotEmpty()
  name: string;

  @IsInt()
  @Min(1)
  size: number;
}

export class CreateDatasetGenerationDto {
  @IsOptional()
  @IsString()
  batchName?: string;

  @IsEnum(ApproachEnum)
  approach: ApproachEnum;

  @ValidateNested({ each: true })
  @Type(() => ChildDatasetSettingsDto)
  datasets: ChildDatasetSettingsDto[];

  @IsOptional()
  @IsInt()
  seed?: number = 1234;
}

export class CreateDatasetGenerationDtoResponse {
  @IsString()
  name: string;

  @IsInt()
  taskId: number;
}
