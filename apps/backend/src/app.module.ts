import { Module } from '@nestjs/common';
import { PrismaModule } from '@prisma/prisma.module';
import { ParentDatasetModule } from '@resources/parent-dataset/parent-dataset.module';
import { ParentRecordModule } from '@resources/parent-record/parent-record.module';
import { ProgressTrackerModule } from '@resources/progress-tracker/progress-tracker.module';
import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [
    PrismaModule,
    ProgressTrackerModule,
    ParentDatasetModule,
    ParentRecordModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
