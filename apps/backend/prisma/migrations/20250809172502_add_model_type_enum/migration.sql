/*
  Warnings:

  - Added the required column `modelType` to the `ChildDataset` table without a default value. This is not possible if the table is not empty.
  - Added the required column `modelType` to the `ParentDataset` table without a default value. This is not possible if the table is not empty.

*/
-- CreateEnum
CREATE TYPE "ModelTypeEnum" AS ENUM ('GPT', 'BERT', 'DNABERT');

-- AlterTable
ALTER TABLE "ChildDataset" ADD COLUMN     "modelType" "ModelTypeEnum" NOT NULL;

-- AlterTable
ALTER TABLE "ParentDataset" ADD COLUMN     "modelType" "ModelTypeEnum" NOT NULL;
