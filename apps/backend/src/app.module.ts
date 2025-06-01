import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { PrismaModule } from './prisma/prisma.module';
import { ProgressTrackerModule } from './progress-tracker/progress-tracker.module';
import { ParentDatasetModule } from './parent-dataset/parent-dataset.module';
import { ParentRecordModule } from './parent-record/parent-record.module';

@Module({
  imports: [PrismaModule, ProgressTrackerModule, ParentDatasetModule, ParentRecordModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
