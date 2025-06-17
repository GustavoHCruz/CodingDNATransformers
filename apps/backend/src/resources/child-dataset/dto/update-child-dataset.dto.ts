import { PartialType } from '@nestjs/swagger';
import { IsOptional, IsString } from 'class-validator';
import { CreateChildDatasetDto } from './create-child-dataset.dto';

export class UpdateChildDatasetDto extends PartialType(CreateChildDatasetDto) {
  @IsOptional()
  @IsString()
  name?: string;
}
