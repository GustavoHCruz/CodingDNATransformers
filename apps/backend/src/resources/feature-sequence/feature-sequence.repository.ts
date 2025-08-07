import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class FeatureSequenceRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.featureSequence.findMany();
  }

  findOne(id: number) {
    return this.prisma.featureSequence.findUnique({ where: { id } });
  }

  create(data: Prisma.FeatureSequenceCreateInput) {
    return this.prisma.featureSequence.create({ data });
  }

  createMany(data: Prisma.FeatureSequenceCreateManyInput[]) {
    return this.prisma.featureSequence.createMany({ data });
  }

  update(id: number, data: Prisma.FeatureSequenceUpdateInput) {
    return this.prisma.featureSequence.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.featureSequence.delete({ where: { id } });
  }

  findExIn(offset, limit, max_length) {
    return this.prisma.featureSequence.findMany({
      select: {
        sequence: true,
        gene: true,
        before: true,
        after: true,
        type: true,
      },
      include: {
        dnaSequence: {
          select: {
            organism: true,
          },
        },
      },
      where: {
        sequence.length < max_length,
        type: {
          equals: ""
        }
      },
    });
  }
}
