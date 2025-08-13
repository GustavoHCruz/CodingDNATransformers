import { Injectable } from '@nestjs/common';
import { FeatureEnum, Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class DnaSequenceRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.dNASequence.findMany();
  }

  findOne(id: number) {
    return this.prisma.dNASequence.findUnique({ where: { id } });
  }

  create(data: Prisma.DNASequenceCreateInput) {
    return this.prisma.dNASequence.create({ data, select: { id: true } });
  }

  async createWithFeatureBatch(
    batch: {
      data: Prisma.DNASequenceCreateInput;
      features: Prisma.FeatureSequenceCreateInput[];
    }[],
  ) {
    const transaction = batch.map((item) =>
      this.prisma.dNASequence.create({
        data: {
          ...item.data,
          features: {
            create: item.features,
          },
        },
      }),
    );

    await this.prisma.$transaction(transaction);
  }

  createMany(data: Prisma.DNASequenceCreateManyInput[]) {
    return this.prisma.dNASequence.createMany({ data });
  }

  update(id: number, data: Prisma.DNASequenceUpdateInput) {
    return this.prisma.dNASequence.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.dNASequence.delete({ where: { id } });
  }

  async findTriplet(maxLength: number, limit: number, lastId: number | null) {
    const results = await this.prisma.dNASequence.findMany({
      where: {
        length: { lt: maxLength },
        ...(lastId !== null && { id: { gt: lastId } }),
      },
      select: {
        id: true,
        sequence: true,
        organism: true,
        length: true,
        features: {
          where: {
            type: {
              in: [FeatureEnum.INTRON, FeatureEnum.EXON],
            },
          },
          select: {
            type: true,
            start: true,
            end: true,
          },
        },
      },
      orderBy: { id: 'asc' },
      take: limit,
    });

    return results;
  }

  async findCDS(maxLength: number, limit: number, lastId: number | null) {
    const results = await this.prisma.$queryRawUnsafe<{
      sequence: string;
      target: string;
      organism: string;
    }>(`
      SELECT
        d.sequence AS sequence,
        f.sequence AS target,
        d.organism AS organism,
        d.id AS id
      FROM "DNASequence" d
      INNER JOIN "FeatureSequence" f 
        ON f."dnaSequenceId" = d.id
        AND f.type = ${FeatureEnum.CDS}
      WHERE d.length < ${maxLength}
        AND (
          SELECT COUNT(*)
          FROM "FeatureSequence" f2
          WHERE f2."dnaSequenceId" = d.id
            AND f2.type = ${FeatureEnum.CDS}
        ) = 1
        ${lastId ? `AND d.id > ${lastId}` : ''}
      ORDER BY d.id ASC
      LIMIT ${limit};
    `);

    return results;
  }
}
