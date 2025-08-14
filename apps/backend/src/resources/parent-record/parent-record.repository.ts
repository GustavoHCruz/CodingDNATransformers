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

  async findByApproach(approach: ApproachEnum, limit = 100) {
    const [intronCount, exonCount] = await Promise.all([
      this.prisma.parentRecord.count({
        where: {
          target: 'INTRON',
          parentDataset: {
            approach,
          },
        },
      }),
      this.prisma.parentRecord.count({
        where: {
          target: 'EXON',
          parentDataset: {
            approach,
          },
        },
      }),
    ]);

    const minCount = Math.min(intronCount, exonCount);

    const [introns, exons] = await Promise.all([
      this.prisma.parentRecord.findMany({
        select: {
          id: true,
        },
        where: {
          target: 'INTRON',
          parentDataset: {
            approach,
          },
        },
        take: minCount,
      }),
      this.prisma.parentRecord.findMany({
        select: {
          id: true,
        },
        where: {
          target: 'EXON',
          parentDataset: {
            approach,
          },
        },
        take: minCount,
      }),
    ]);

    return [...introns, ...exons];
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
    return this.prisma.parentRecord.createMany({ data, skipDuplicates: true });
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

  findByParentIdPaginated(
    parentId: number,
    cursorId?: number | null,
    limit = 100,
  ) {
    return this.prisma.parentRecord.findMany({
      where: {
        parentDatasetId: parentId,
      },
      select: {
        id: true,
        sequence: true,
        target: true,
        organism: true,
        gene: true,
        flankBefore: true,
        flankAfter: true,
      },
      take: limit,
      skip: cursorId ? 1 : 0,
      cursor: cursorId ? { id: cursorId } : undefined,
      orderBy: {
        id: 'asc',
      },
    });
  }
}
