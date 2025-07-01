import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class ModelHistoryRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.modelHistory.findMany();
  }

  findOne(id: number) {
    return this.prisma.modelHistory.findUnique({ where: { id } });
  }

  create(data: Prisma.ModelHistoryCreateInput) {
    return this.prisma.modelHistory.create({ data });
  }

  update(id: number, data: Prisma.ModelHistoryUpdateInput) {
    return this.prisma.modelHistory.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.modelHistory.delete({ where: { id } });
  }
}
