import { Module } from '@nestjs/common';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import { ExtractionGrpcClientService } from './extraction.grpc-client.service';

@Module({
  imports: [
    ClientsModule.register([
      {
        name: 'EXTRACTION_PACKAGE',
        transport: Transport.GRPC,
        options: {
          package: 'extraction',
          protoPath: join(__dirname, 'proto/extraction.proto'),
          url: 'localhost:50051',
        },
      },
    ]),
  ],
  providers: [ExtractionGrpcClientService],
  exports: [ExtractionGrpcClientService],
})
export class ExtractionGrpcClientModule {}
