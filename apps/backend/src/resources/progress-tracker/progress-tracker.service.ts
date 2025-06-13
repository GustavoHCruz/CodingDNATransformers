import { Injectable } from '@nestjs/common';
import { CreateProgressTrackerDto } from './dto/create-progress-tracker.dto';
import { ProgressTrackerRepository } from './progress-tracker.repository';

@Injectable()
export class ProgressTrackerService {
  constructor(private repository: ProgressTrackerRepository) {}

  create(createProgressTrackerDto: CreateProgressTrackerDto) {
    return this.repository.create(createProgressTrackerDto);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  setProgress(id: number, progress: number) {
    return this.repository.update(id, {
      progress,
    });
  }

  remove(id: number) {
    return this.repository.remove(id);
  }

  async postPogress(taskId: number, counter: number, total?: number) {
    let progress = counter;
    if (total) {
      progress = (counter * 100) / total;
    }

    await this.setProgress(taskId, progress);
  }

  finish(taskId: number) {
    return this.repository.update(taskId, {
      status: 'COMPLETE',
    });
  }
}
