import { PartialType } from '@nestjs/swagger';
import { IsString } from 'class-validator';
import { CreateGenerationBatchDto } from './create-generation-batch.dto';

export class UpdateGenerationBatchDto extends PartialType(
  CreateGenerationBatchDto,
) {
  @IsString()
  name: string;
}
