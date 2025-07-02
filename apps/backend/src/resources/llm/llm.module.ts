import { Module } from '@nestjs/common';
import { ChildRecordModule } from '@resources/child-record/child-record.module';
import { LlmController } from './llm.controller';
import { LlmService } from './llm.service';

@Module({
  imports: [ChildRecordModule],
  controllers: [LlmController],
  providers: [LlmService],
  exports: [LlmService],
})
export class LlmModule {}
