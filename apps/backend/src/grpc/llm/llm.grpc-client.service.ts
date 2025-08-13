import { Inject, Injectable, OnModuleInit } from '@nestjs/common';
import { ClientGrpc } from '@nestjs/microservices';
import { LlmRequest, LlmsService } from './interfaces/llm.interface';

@Injectable()
export class LlmGrpcClientService implements OnModuleInit {
  private llmService: LlmsService;

  constructor(@Inject('LLM_PACKAGE') private client: ClientGrpc) {}

  onModuleInit() {
    this.llmService = this.client.getService<LlmsService>('LlmService');
  }

  call(method: keyof LlmsService, req: LlmRequest): null {
    return this.llmService[method](req);
  }
}
