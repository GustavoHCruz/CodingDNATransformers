import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import {
  IsBoolean,
  IsNumber,
  IsOptional,
  ValidateNested,
} from 'class-validator';

class ExInClassifier {
  @IsBoolean()
  active: boolean = false;

  @IsBoolean()
  gpt: boolean = false;

  @IsBoolean()
  bert: boolean = false;

  @IsBoolean()
  dnabert: boolean = false;
}

class TripletClassifier {
  @IsBoolean()
  active: boolean = false;

  @IsBoolean()
  bert: boolean = false;

  @IsBoolean()
  dnabert: boolean = false;
}

class DNATranslator {
  @IsBoolean()
  active: boolean = false;

  @IsBoolean()
  gpt: boolean = false;

  @IsBoolean()
  alternative: boolean = false;
}

class ExInClassifierResponse {
  @IsNumber()
  @IsOptional()
  gpt?: number;

  @IsNumber()
  @IsOptional()
  bert?: number;

  @IsNumber()
  @IsOptional()
  dnabert?: number;
}

class TripletClassifierResponse {
  @IsNumber()
  @IsOptional()
  bert?: number;

  @IsNumber()
  @IsOptional()
  dnabert?: number;
}

class DNATranslatorResponse {
  @IsNumber()
  @IsOptional()
  gpt?: number;

  @IsNumber()
  @IsOptional()
  alternative?: number;
}

class Approach {
  @ValidateNested()
  @Type(() => ExInClassifier)
  @IsOptional()
  ExInClassifier?: ExInClassifier;

  @ValidateNested()
  @Type(() => TripletClassifier)
  @IsOptional()
  TripletClassifier?: TripletClassifier;

  @ValidateNested()
  @Type(() => DNATranslator)
  @IsOptional()
  DNATranslator?: DNATranslator;
}

class ApproachResponse {
  @ApiProperty()
  @ValidateNested()
  @Type(() => ExInClassifierResponse)
  @IsOptional()
  ExInClassifier?: ExInClassifierResponse;

  @ApiProperty()
  @ValidateNested()
  @Type(() => TripletClassifierResponse)
  @IsOptional()
  TripletClassifier?: TripletClassifierResponse;

  @ApiProperty()
  @ValidateNested()
  @Type(() => DNATranslatorResponse)
  @IsOptional()
  DNATranslator?: DNATranslatorResponse;
}

export class CreateRawDatasetsDto {
  @ApiProperty()
  @IsOptional()
  @ValidateNested()
  @Type(() => Approach)
  genbank?: Approach;
}

export class CreateRawDatasetsDtoResponse {
  @ApiProperty()
  @ValidateNested()
  @Type(() => ApproachResponse)
  genbank: ApproachResponse = new ApproachResponse();
}
