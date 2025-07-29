/*
  Warnings:

  - The values [EXINTRANSLATOR] on the enum `ApproachEnum` will be removed. If these variants are still used in the database, this will fail.

*/
-- AlterEnum
BEGIN;
CREATE TYPE "ApproachEnum_new" AS ENUM ('EXINCLASSIFIER', 'SLIDINGWINDOWEXTRACTION', 'PROTEINTRANSLATOR');
ALTER TABLE "RawFileInfo" ALTER COLUMN "approach" TYPE "ApproachEnum_new" USING ("approach"::text::"ApproachEnum_new");
ALTER TABLE "ParentDataset" ALTER COLUMN "approach" TYPE "ApproachEnum_new" USING ("approach"::text::"ApproachEnum_new");
ALTER TABLE "ChildDataset" ALTER COLUMN "approach" TYPE "ApproachEnum_new" USING ("approach"::text::"ApproachEnum_new");
ALTER TABLE "ModelHistory" ALTER COLUMN "approach" TYPE "ApproachEnum_new" USING ("approach"::text::"ApproachEnum_new");
ALTER TYPE "ApproachEnum" RENAME TO "ApproachEnum_old";
ALTER TYPE "ApproachEnum_new" RENAME TO "ApproachEnum";
DROP TYPE "ApproachEnum_old";
COMMIT;
