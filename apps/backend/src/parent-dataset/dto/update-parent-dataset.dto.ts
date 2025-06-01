import { PartialType } from '@nestjs/swagger';
import { IsOptional, IsString } from 'class-validator';
import { CreateParentDatasetDto } from './create-parent-dataset.dto';

export class UpdateParentDatasetDto extends PartialType(
  CreateParentDatasetDto,
) {
  @IsOptional()
  @IsString()
  name?: string;
}
