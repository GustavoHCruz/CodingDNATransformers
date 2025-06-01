import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from 'src/prisma/prisma.service';

@Injectable()
export class ParentRecordRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.parentRecord.findMany();
  }

  findOne(id: number) {
    return this.prisma.parentRecord.findUnique({ where: { id } });
  }

  create(data: Prisma.ParentRecordCreateInput) {
    return this.prisma.parentRecord.create({ data });
  }

  update(id: number, data: Prisma.ParentRecordUpdateInput) {
    return this.prisma.parentRecord.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.parentRecord.delete({ where: { id } });
  }
}
