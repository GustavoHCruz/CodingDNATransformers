import { IsInt, IsNumber } from 'class-validator';

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

  @IsNumber()
  loss: number;

  @IsInt()
  durationSec: number;

  @IsInt()
  seed: number;
}
