export interface CreateModelRequest {
  uuid: string;
  modelAlias: string;
  checkpointName: string;
}

export interface TrainModelRequest {
  uuid: string;
  modelAlias: string;
  learningRate: number;
  batchSize: number;
  gradientAccumulation: number;
  warmupRatio: number;
  epochs: number;
  hideProb?: number;
  lora: boolean;
  seed: number;
}

export interface EvalModelRequest {
  uuid: string;
  modelAlias: string;
}

export interface PredictRequest {
  uuid: string;
  inputText: string;
}
export type LlmRequest = CreateModelRequest &
  TrainModelRequest &
  EvalModelRequest &
  PredictRequest;

export interface LlmsService {
  createModel(req: CreateModelRequest): null;
  trainModel(req: TrainModelRequest): null;
  evalModel(req: EvalModelRequest): null;
  predict(req: PredictRequest): null;
}
