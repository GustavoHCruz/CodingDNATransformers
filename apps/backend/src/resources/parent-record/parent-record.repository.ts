import { Injectable } from '@nestjs/common';
import { ApproachEnum, ParentRecord, Prisma } from '@prisma/client';
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
      select: { id: true },
      where: {
        parentDataset: {
          approach,
        },
      },
      take: limit,
    });
  }

  async findByApproachInBatches(
    approach: ApproachEnum,
    total: number,
    batchSize = 100000,
  ): Promise<ParentRecord[]> {
    const results: ParentRecord[] = [];
    let lastId: number | null = null;

    while (results.length < total) {
      const batch = await this.prisma.parentRecord.findMany({
        select: { id: true },
        where: {
          parentDataset: {
            approach,
          },
          ...(lastId && {
            id: { gt: lastId },
          }),
        },
        orderBy: { id: 'asc' },
        take: Math.min(batchSize, total - results.length),
      });

      if (batch.length === 0) {
        break;
      }

      results.push(...batch);
      lastId = batch[batch.length - 1].id;
    }

    return results;
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

  countByApproach(approach: ApproachEnum) {
    return this.prisma.parentRecord.count({
      where: {
        parentDataset: {
          approach,
        },
      },
    });
  }
}
