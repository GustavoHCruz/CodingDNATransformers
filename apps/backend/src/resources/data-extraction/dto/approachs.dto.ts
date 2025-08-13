import { IsNumber } from 'class-validator';

export class DataExtractionReturn {
  @IsNumber()
  taskId: number;
}
