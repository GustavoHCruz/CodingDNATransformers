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
    const results = await this.prisma.$queryRawUnsafe<
      {
        id: number;
        sequence: string;
        gene: string;
        before: string;
        after: string;
        type: string;
        organism: string;
      }[]
    >(`
SELECT
  f.id,
  f.sequence,
  f.gene,
  f.before,
  f.after,
  f.type,
  d.organism
FROM "FeatureSequence" f
JOIN "DNASequence" d ON f."dnaSequenceId" = d.id
WHERE LENGTH(f.sequence) < ${maxLength}
  AND f.type in ('${FeatureEnum.EXON}','${FeatureEnum.INTRON}')
  ${lastId !== null ? `AND f.id > ${lastId}` : ``}
  ORDER BY f.id
  LIMIT ${limit}
`);
    return results;
  }

  async findCDS(maxLength: number) {
    const results = await this.prisma.$queryRaw<
      {
        sequence: string;
        protein: string;
        gene: string;
        before: string;
        after: string;
        organism: string;
      }[]
    >`
    SELECT
      d.sequence as sequence,
      f.sequence as protein,
      f.gene as gene,
      f.before as before,
      f.after as after,
      d.organism as organism
    FROM "FeatureSequence" f
    JOIN "DNASequence" d ON f."dnaSequenceId" = d.id
    WHERE LENGTH(d.sequence) < ${maxLength}
      AND f.type = ${FeatureEnum.CDS}
      AND (
        SELECT COUNT(*)
        FROM "FeatureSequence" f2
        WHERE f2."dnaSequenceId" = f."dnaSequenceId"
          AND f2.type = ${FeatureEnum.CDS}
      ) = 1
  `;
    return results;
  }
}
