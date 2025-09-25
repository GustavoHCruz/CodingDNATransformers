import { Injectable } from '@nestjs/common';
import { FeatureEnum, Prisma } from '@prisma/client';
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

  async findExIn(maxLength: number, limit: number, lastId: number | null) {
    const results = await this.prisma.featureSequence.findMany({
      where: {
        length: { lt: maxLength },
        type: { in: [FeatureEnum.EXON, FeatureEnum.INTRON] },
        ...(lastId !== null && { id: { gt: lastId } }),
      },
      select: {
        id: true,
        sequence: true,
        gene: true,
        before: true,
        after: true,
        type: true,
        dnaSequence: {
          select: { organism: true },
        },
      },
      orderBy: { id: 'asc' },
      take: limit,
    });

    return results;
  }

  async findWithCDS(
    dnaMaxLength: number,
    proteinMaxLength: number,
    limit: number,
    lastId: number | null,
  ) {
    const withCDS = await this.prisma.dNASequence.findMany({
      select: {
        id: true,
        sequence: true,
        organism: true,
        length: true,
        features: {
          select: {
            id: true,
            sequence: true,
            length: true,
          },
          orderBy: { start: 'asc' },
        },
      },
      where: {
        ...(lastId !== null && { id: { gt: lastId } }),
        length: { lte: dnaMaxLength },
        features: {
          some: {
            type: FeatureEnum.CDS,
            length: {
              lte: proteinMaxLength,
            },
          },
        },
      },
      orderBy: { id: 'asc' },
      take: limit,
    });

    return withCDS;
  }
}
