import { Injectable } from '@nestjs/common';
import { ApproachEnum, Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class ParentRecordRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.parentRecord.findMany();
  }

  findOne(id: number) {
    return this.prisma.parentRecord.findUnique({ where: { id } });
  }

  findByApproach(approach: ApproachEnum, limit = 100) {
    return this.prisma.parentRecord.findMany({
      where: {
        parentDataset: {
          approach,
        },
      },
      take: limit,
    });
  }

  create(data: Prisma.ParentRecordCreateInput) {
    return this.prisma.parentRecord.create({ data });
  }

  createMany(data: Prisma.ParentRecordCreateManyInput[]) {
    return this.prisma.parentRecord.createMany({ data });
  }

  update(id: number, data: Prisma.ParentRecordUpdateInput) {
    return this.prisma.parentRecord.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.parentRecord.delete({ where: { id } });
  }
}
