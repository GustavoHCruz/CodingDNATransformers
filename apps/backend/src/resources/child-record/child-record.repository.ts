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

  findByChildIdPaginated(
    childId: number,
    cursorId?: number | null,
    limit = 100,
  ) {
    return this.prisma.childRecord.findMany({
      where: {
        childDatasetId: childId,
      },
      include: {
        parentRecord: {
          select: {
            id: true,
            sequence: true,
            target: true,
            organism: true,
            gene: true,
            flankBefore: true,
            flankAfter: true,
          },
        },
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
