import { Module } from '@nestjs/common';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { SHARED_DIR } from 'common/constrants';
import { join } from 'path';

@Module({
  imports: [
    ClientsModule.register([
      {
        name: 'LLM_PACKAGE',
        transport: Transport.GRPC,
        options: {
          package: 'llm',
          protoPath: join(SHARED_DIR, 'protos/llm.proto'),
          url: process.env.LLMS_SERVICE_URL || 'localhost:50051',
        },
      },
    ]),
  ],
})
export class LlmGrpcClientModule {}
