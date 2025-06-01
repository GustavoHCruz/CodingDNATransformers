import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from 'src/prisma/prisma.service';

@Injectable()
export class ProgressTrackerRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.progressTracker.findMany();
  }

  findOne(id: number) {
    return this.prisma.progressTracker.findUnique({ where: { id } });
  }

  create(data: Prisma.ProgressTrackerCreateInput) {
    return this.prisma.progressTracker.create({ data });
  }

  update(id: number, data: Prisma.ProgressTrackerUpdateInput) {
    return this.prisma.progressTracker.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.progressTracker.delete({ where: { id } });
  }
}
