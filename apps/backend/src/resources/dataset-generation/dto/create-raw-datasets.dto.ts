import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import { IsBoolean, IsInt, IsOptional, ValidateNested } from 'class-validator';

class ApproachDto {
  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  ExInClassifier?: boolean = false;

  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  SlidingWindowTagger?: boolean = false;

  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  ProteinTranslator?: boolean = false;
}

class ApproachResponseDto {
  @ApiProperty()
  @IsInt()
  @IsOptional()
  ExInClassifier?: number;

  @ApiProperty()
  @IsInt()
  @IsOptional()
  SlidingWindowTagger?: number;

  @ApiProperty()
  @IsInt()
  @IsOptional()
  ProteinTranslator?: number;
}

export class CreateRawDatasetsDto {
  @ApiProperty()
  @IsOptional()
  @ValidateNested()
  @Type(() => ApproachDto)
  genbank?: ApproachDto;
}

export class CreateRawDatasetsDtoResponse {
  @ApiProperty()
  @ValidateNested()
  @Type(() => ApproachResponseDto)
  genbank: ApproachResponseDto = new ApproachResponseDto();
}
