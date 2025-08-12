import { CreateFeatureSequenceDto } from '@resources/feature-sequence/dto/create-feature-sequence.dto';
import { IsNumber, IsOptional, IsString } from 'class-validator';

export class CreateDnaSequenceDto {
  @IsString()
  sequence: string;

  @IsNumber()
  length: number;

  @IsOptional()
  @IsString()
  accession?: string;

  @IsOptional()
  @IsString()
  organism?: string;
}

export class CreateDnaSequenceWithFeatureBatchDto {
  data: CreateDnaSequenceDto;
  features: Omit<CreateFeatureSequenceDto, 'dnaSequenceId'>[];
}
