import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class GenerationBatchRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.generationBatch.findMany();
  }

  findOne(id: number) {
    return this.prisma.generationBatch.findUnique({ where: { id } });
  }

  create(data: Prisma.GenerationBatchCreateInput) {
    return this.prisma.generationBatch.create({ data });
  }

  createMany(data: Prisma.GenerationBatchCreateManyInput[]) {
    return this.prisma.generationBatch.createMany({ data });
  }

  update(id: number, data: Prisma.GenerationBatchUpdateInput) {
    return this.prisma.generationBatch.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.generationBatch.delete({ where: { id } });
  }
}
