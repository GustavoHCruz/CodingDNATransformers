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
}
