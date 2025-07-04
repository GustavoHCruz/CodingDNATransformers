import { Inject, Injectable, OnModuleInit } from '@nestjs/common';
import { ClientGrpc } from '@nestjs/microservices';
import { Observable } from 'rxjs';
import {
  ExtractionRequest,
  ExtractionResponse,
  ExtractionService,
} from './interfaces/extraction.interface';

@Injectable()
export class ExtractionGrpcClientService implements OnModuleInit {
  private extractionService: ExtractionService;

  constructor(@Inject('EXTRACTION_PACKAGE') private client: ClientGrpc) {}

  onModuleInit() {
    this.extractionService =
      this.client.getService<ExtractionService>('ExtractionService');
  }

  call(
    method: keyof ExtractionService,
    req: ExtractionRequest,
  ): Observable<ExtractionResponse> {
    return this.extractionService[method](req);
  }
}
