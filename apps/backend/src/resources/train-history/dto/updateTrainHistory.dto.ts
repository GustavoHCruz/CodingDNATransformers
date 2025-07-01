import { PartialType } from '@nestjs/swagger';
import { CreateTrainHistoryDto } from './createTrainHistory.dto';

export class UpdateTrainHistoryDto extends PartialType(CreateTrainHistoryDto) {}
