import { ConfigModule } from '@config/config.module';
import { Module } from '@nestjs/common';
import { ParentRecordModule } from '@resources/parent-record/parent-record.module';
import { LlmController } from './llm.controller';
import { LlmService } from './llm.service';

@Module({
  imports: [ConfigModule, ParentRecordModule],
  controllers: [LlmController],
  providers: [LlmService],
  exports: [LlmService],
})
export class LlmModule {}
