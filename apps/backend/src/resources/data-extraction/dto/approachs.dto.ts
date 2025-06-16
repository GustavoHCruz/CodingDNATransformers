import { ApiProperty } from '@nestjs/swagger';
import { IsBoolean, IsInt, IsOptional } from 'class-validator';

export class ApproachDto {
  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  ExInClassifier?: boolean = false;

  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  ExInTranslator?: boolean = false;

  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  SlidingWindowTagger?: boolean = false;

  @ApiProperty()
  @IsBoolean()
  @IsOptional()
  ProteinTranslator?: boolean = false;
}

export class ApproachResponseDto {
  @ApiProperty()
  @IsInt()
  @IsOptional()
  ExInClassifier?: number;

  @ApiProperty()
  @IsInt()
  @IsOptional()
  ExInTranslator?: number;

  @ApiProperty()
  @IsInt()
  @IsOptional()
  SlidingWindowTagger?: number;

  @ApiProperty()
  @IsInt()
  @IsOptional()
  ProteinTranslator?: number;
}
