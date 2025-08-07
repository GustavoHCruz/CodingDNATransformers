import { Injectable } from '@nestjs/common';
import { OriginEnum, Prisma } from '@prisma/client';
import { PrismaService } from '@prisma/prisma.service';

@Injectable()
export class RawFileInfoRepository {
  constructor(private prisma: PrismaService) {}

  findAll() {
    return this.prisma.rawFileInfo.findMany();
  }

  findOne(id: number) {
    return this.prisma.rawFileInfo.findUnique({ where: { id } });
  }

  findOneByFileName(fileName: string) {
    return this.prisma.rawFileInfo.findFirst({
      where: { fileName },
    });
  }

  findOneByFileNameAndApproach(fileName: string, origin: OriginEnum) {
    return this.prisma.rawFileInfo.findFirst({
      where: {
        fileName,
        origin,
      },
    });
  }

  create(data: Prisma.RawFileInfoCreateInput) {
    return this.prisma.rawFileInfo.create({ data });
  }

  update(id: number, data: Prisma.RawFileInfoUpdateInput) {
    return this.prisma.rawFileInfo.update({ where: { id }, data });
  }

  remove(id: number) {
    return this.prisma.rawFileInfo.delete({ where: { id } });
  }
}
