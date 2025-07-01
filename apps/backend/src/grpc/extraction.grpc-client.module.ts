import { Module } from '@nestjs/common';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { SHARED_DIR } from 'common/constrants';
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
          protoPath: join(SHARED_DIR, 'protos/data.proto'),
          url: 'localhost:50051',
        },
      },
    ]),
  ],
  providers: [ExtractionGrpcClientService],
  exports: [ExtractionGrpcClientService],
})
export class ExtractionGrpcClientModule {}
