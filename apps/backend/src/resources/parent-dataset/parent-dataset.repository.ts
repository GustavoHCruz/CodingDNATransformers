import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class ParentDatasetRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.parentDataset.findMany();
  }

  findOne(id: number) {
    return this.prisma.parentDataset.findUnique({ where: { id } });
  }

  create(data: Prisma.ParentDatasetCreateInput) {
    return this.prisma.parentDataset.create({ data });
  }

  update(id: number, data: Prisma.ParentDatasetUpdateInput) {
    return this.prisma.parentDataset.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.parentDataset.delete({ where: { id } });
  }
}
