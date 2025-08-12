/*
  Warnings:

  - Added the required column `length` to the `DNASequence` table without a default value. This is not possible if the table is not empty.
  - Added the required column `length` to the `FeatureSequence` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "DNASequence" ADD COLUMN     "length" INTEGER NOT NULL;

-- AlterTable
ALTER TABLE "FeatureSequence" ADD COLUMN     "length" INTEGER NOT NULL;
