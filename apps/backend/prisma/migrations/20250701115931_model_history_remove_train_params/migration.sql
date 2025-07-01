/*
  Warnings:

  - You are about to drop the column `accuracy` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `batchSize` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `durationSec` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `epochs` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `hideProp` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `learningRate` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `seed` on the `ModelHistory` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "ModelHistory" DROP COLUMN "accuracy",
DROP COLUMN "batchSize",
DROP COLUMN "durationSec",
DROP COLUMN "epochs",
DROP COLUMN "hideProp",
DROP COLUMN "learningRate",
DROP COLUMN "seed";
