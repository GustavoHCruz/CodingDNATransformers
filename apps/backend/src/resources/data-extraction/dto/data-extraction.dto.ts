import { Type } from 'class-transformer';
import { IsOptional, ValidateNested } from 'class-validator';
import { ApproachDto, ApproachResponseDto } from './approachs.dto';

export class CreateDataExtractionDto {
  @IsOptional()
  @ValidateNested()
  @Type(() => ApproachDto)
  genbank?: ApproachDto;

  @IsOptional()
  @ValidateNested()
  @Type(() => ApproachDto)
  gencode?: ApproachDto;
}

export class CreateDataExtractionResponseDto {
  @ValidateNested()
  @Type(() => ApproachResponseDto)
  genbank: ApproachResponseDto = new ApproachResponseDto();

  @ValidateNested()
  @Type(() => ApproachResponseDto)
  gencode: ApproachResponseDto = new ApproachResponseDto();
}
