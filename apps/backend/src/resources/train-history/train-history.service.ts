import { Injectable } from '@nestjs/common';
import { CreateTrainHistoryDto } from './dto/createTrainHistory.dto';
import { UpdateTrainHistoryDto } from './dto/updateTrainHistory.dto';
import { TrainHistoryRepository } from './train-history.repository';

@Injectable()
export class TrainHistoryService {
  constructor(private repository: TrainHistoryRepository) {}

  create(data: CreateTrainHistoryDto) {
    return this.repository.create(data);
  }

  findAll() {
    return this.repository.findAll();
  }

  findOne(id: number) {
    return this.repository.findOne(id);
  }

  update(id: number, data: UpdateTrainHistoryDto) {
    return this.repository.update(id, data);
  }

  remove(id: number) {
    return this.repository.remove(id);
  }
}
