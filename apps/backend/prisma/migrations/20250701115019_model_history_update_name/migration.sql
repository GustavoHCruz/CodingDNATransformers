/*
  Warnings:

  - You are about to drop the column `modelName` on the `ModelHistory` table. All the data in the column will be lost.
  - You are about to drop the column `name` on the `ModelHistory` table. All the data in the column will be lost.
  - Added the required column `checkpointName` to the `ModelHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `modelAlias` to the `ModelHistory` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "ModelHistory" DROP COLUMN "modelName",
DROP COLUMN "name",
ADD COLUMN     "checkpointName" TEXT NOT NULL,
ADD COLUMN     "modelAlias" TEXT NOT NULL;
