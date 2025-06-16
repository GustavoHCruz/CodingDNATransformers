import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import { IsOptional, ValidateNested } from 'class-validator';
import { ApproachDto, ApproachResponseDto } from './approachs.dto';

export class CreateDataExtractionDto {
  @ApiProperty()
  @IsOptional()
  @ValidateNested()
  @Type(() => ApproachDto)
  genbank?: ApproachDto;

  @ApiProperty()
  @IsOptional()
  @ValidateNested()
  @Type(() => ApproachDto)
  gencode?: ApproachDto;
}

export class CreateDataExtractionResponseDto {
  @ApiProperty()
  @ValidateNested()
  @Type(() => ApproachResponseDto)
  genbank: ApproachResponseDto = new ApproachResponseDto();

  @ApiProperty()
  @ValidateNested()
  @Type(() => ApproachResponseDto)
  gencode: ApproachResponseDto = new ApproachResponseDto();
}
