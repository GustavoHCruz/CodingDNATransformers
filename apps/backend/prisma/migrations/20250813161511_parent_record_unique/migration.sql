/*
  Warnings:

  - A unique constraint covering the columns `[sequence,target,flankBefore,flankAfter,organism,gene,parentDatasetId]` on the table `ParentRecord` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterTable
ALTER TABLE "ParentRecord" ALTER COLUMN "flankBefore" SET DEFAULT '',
ALTER COLUMN "flankAfter" SET DEFAULT '',
ALTER COLUMN "organism" SET DEFAULT '',
ALTER COLUMN "gene" SET DEFAULT '';

-- CreateIndex
CREATE UNIQUE INDEX "ParentRecord_sequence_target_flankBefore_flankAfter_organis_key" ON "ParentRecord"("sequence", "target", "flankBefore", "flankAfter", "organism", "gene", "parentDatasetId");
