import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class ChildDatasetRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.childDataset.findMany();
  }

  findOne(id: number) {
    return this.prisma.childDataset.findUnique({ where: { id } });
  }

  create(data: Prisma.ChildDatasetCreateInput) {
    return this.prisma.childDataset.create({ data });
  }

  update(id: number, data: Prisma.ChildDatasetUpdateInput) {
    return this.prisma.childDataset.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.childDataset.delete({ where: { id } });
  }
}
