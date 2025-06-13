import { Module } from '@nestjs/common';
import { RawFileInfoController } from './raw-file-info.controller';
import { RawFileInfoRepository } from './raw-file-info.repository';
import { RawFileInfoService } from './raw-file-info.service';

@Module({
  controllers: [RawFileInfoController],
  providers: [RawFileInfoRepository, RawFileInfoService],
})
export class RawFileInfoModule {}
