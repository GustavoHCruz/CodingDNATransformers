import { ConfigModule } from '@config/config.module';
import { Module } from '@nestjs/common';
import { ChildRecordModule } from '@resources/child-record/child-record.module';
import { ParentRecordModule } from '@resources/parent-record/parent-record.module';
import { LlmController } from './llm.controller';
import { LlmService } from './llm.service';

@Module({
  imports: [ConfigModule, ParentRecordModule, ChildRecordModule],
  controllers: [LlmController],
  providers: [LlmService],
  exports: [LlmService],
})
export class LlmModule {}
