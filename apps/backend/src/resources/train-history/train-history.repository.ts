import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class TrainHistoryRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.trainHistory.findMany();
  }

  findOne(id: number) {
    return this.prisma.trainHistory.findUnique({ where: { id } });
  }

  create(data: Prisma.TrainHistoryCreateInput) {
    return this.prisma.trainHistory.create({ data });
  }

  update(id: number, data: Prisma.TrainHistoryUpdateInput) {
    return this.prisma.trainHistory.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.trainHistory.delete({ where: { id } });
  }
}
