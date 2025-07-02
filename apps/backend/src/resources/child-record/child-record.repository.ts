import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class ChildRecordRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.childRecord.findMany();
  }

  findOne(id: number) {
    return this.prisma.childRecord.findUnique({ where: { id } });
  }

  async *streamFindAllByChildDatasetId(
    childDatasetId: number,
    chunkSize = 1000,
  ) {
    let lastId: number | null = null;
    let hasMore = true;

    while (hasMore) {
      const records = await this.prisma.childRecord.findMany({
        where: {
          childDatasetId,
          ...(lastId ? { id: { gt: lastId } } : {}),
        },
        take: chunkSize,
        orderBy: {
          id: 'asc',
        },
        include: {
          parentRecord: {
            select: {
              sequence: true,
              target: true,
              organism: true,
              gene: true,
              flankBefore: true,
              flankAfter: true,
            },
          },
        },
      });

      for (const record of records) {
        lastId = record.id;
        if (record.parentRecord) {
          yield record.parentRecord;
        }
      }

      hasMore = records.length === chunkSize;
    }
  }

  create(data: Prisma.ChildRecordUncheckedCreateInput) {
    return this.prisma.childRecord.create({ data });
  }

  createMany(data: Prisma.ChildRecordCreateManyInput[]) {
    return this.prisma.childRecord.createMany({ data });
  }

  update(id: number, data: Prisma.ChildRecordUpdateInput) {
    return this.prisma.childRecord.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.childRecord.delete({ where: { id } });
  }
}
