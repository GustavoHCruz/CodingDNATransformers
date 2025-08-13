import { IsString } from 'class-validator';

export class CreateGenerationBatchDto {
  @IsString()
  name: string;
}
