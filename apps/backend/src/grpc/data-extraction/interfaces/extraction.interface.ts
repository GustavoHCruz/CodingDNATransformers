import { Observable } from 'rxjs';

export interface ExtractionRequest {
  path: string;
}

interface ExInRegion {
  sequence: string;
  type: 'EXON' | 'INTRON';
  start: number;
  end: number;
  gene?: string;
  strand?: number;
  before?: string;
  after?: string;
}

interface CDSRegion {
  sequence: string;
  type: 'CDS';
  start: number;
  end: number;
  gene?: string;
}

export interface ExtractionResponse {
  sequence: string;
  accession: string;
  organism: string;
  cds: CDSRegion[];
  exin: ExInRegion[];
}

export interface ExtractionService {
  Extract(req: ExtractionRequest): Observable<ExtractionResponse>;
}
