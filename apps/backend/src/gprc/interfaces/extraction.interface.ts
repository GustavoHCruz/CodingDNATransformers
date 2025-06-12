import { Observable } from 'rxjs';

export interface ExtractionService {
  ExInClassifierGenbank(req: ExtractionRequest): Observable<ExtractionResponse>;
  ExInTranslatorGenbank(req: ExtractionRequest): Observable<ExtractionResponse>;
  SlidingWindowTaggerGenbank(
    req: ExtractionRequest,
  ): Observable<ExtractionResponse>;
  ProteinTranslatorGenbank(
    req: ExtractionRequest,
  ): Observable<ExtractionResponse>;
  ExInClassifierGencode(req: ExtractionRequest): Observable<ExtractionResponse>;
  ExInTranslatorGencode(req: ExtractionRequest): Observable<ExtractionResponse>;
  SlidingWindowTaggerGencode(
    req: ExtractionRequest,
  ): Observable<ExtractionResponse>;
  ProteinTranslatorGencode(
    req: ExtractionRequest,
  ): Observable<ExtractionResponse>;
}

export interface ExtractionRequest {
  parentId: number;
  sequenceMaxLength: number;
}

export interface ExtractionResponse {
  parentId: number;
  sequence: string;
  target: string;
  flankBefore: string;
  flankAfter: string;
  organism: string;
  gene: string;
}
