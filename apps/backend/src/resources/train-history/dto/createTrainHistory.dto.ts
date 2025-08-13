import { IsBoolean, IsInt, IsNumber, IsOptional } from 'class-validator';

export class CreateTrainHistoryDto {
  @IsInt()
  learningRate: number;

  @IsInt()
  batchSize: number;

  @IsInt()
  gradientAccumulation: number;

  @IsNumber()
  warmupRatio: number;

  @IsInt()
  epochs: number;

  @IsNumber()
  hideProb: number;

  @IsOptional()
  @IsBoolean()
  lora?: boolean;

  @IsNumber()
  loss: number;

  @IsInt()
  durationSec: number;

  @IsInt()
  seed: number;
}
