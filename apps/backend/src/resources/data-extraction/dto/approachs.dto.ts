import { IsBoolean, IsInt, IsOptional } from 'class-validator';

export class ApproachDto {
  @IsBoolean()
  @IsOptional()
  ExInClassifier?: boolean = false;

  @IsBoolean()
  @IsOptional()
  ExInTranslator?: boolean = false;

  @IsBoolean()
  @IsOptional()
  SlidingWindowTagger?: boolean = false;

  @IsBoolean()
  @IsOptional()
  ProteinTranslator?: boolean = false;
}

export class ApproachResponseDto {
  @IsInt()
  @IsOptional()
  ExInClassifier?: number;

  @IsInt()
  @IsOptional()
  ExInTranslator?: number;

  @IsInt()
  @IsOptional()
  SlidingWindowTagger?: number;

  @IsInt()
  @IsOptional()
  ProteinTranslator?: number;
}
