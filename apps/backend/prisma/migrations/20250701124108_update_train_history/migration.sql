/*
  Warnings:

  - You are about to drop the column `epoch` on the `TrainHistory` table. All the data in the column will be lost.
  - Added the required column `batchSize` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `epochs` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `gradientAccumulation` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `hideProb` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `learningRate` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `seed` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.
  - Added the required column `warmupRatio` to the `TrainHistory` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "TrainHistory" DROP COLUMN "epoch",
ADD COLUMN     "batchSize" INTEGER NOT NULL,
ADD COLUMN     "epochs" INTEGER NOT NULL,
ADD COLUMN     "evalDatasetId" INTEGER,
ADD COLUMN     "gradientAccumulation" INTEGER NOT NULL,
ADD COLUMN     "hideProb" DOUBLE PRECISION NOT NULL,
ADD COLUMN     "learningRate" INTEGER NOT NULL,
ADD COLUMN     "seed" INTEGER NOT NULL,
ADD COLUMN     "trainDatasetId" INTEGER,
ADD COLUMN     "warmupRatio" DOUBLE PRECISION NOT NULL;

-- AddForeignKey
ALTER TABLE "TrainHistory" ADD CONSTRAINT "TrainHistory_trainDatasetId_fkey" FOREIGN KEY ("trainDatasetId") REFERENCES "ChildDataset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TrainHistory" ADD CONSTRAINT "TrainHistory_evalDatasetId_fkey" FOREIGN KEY ("evalDatasetId") REFERENCES "ChildDataset"("id") ON DELETE SET NULL ON UPDATE CASCADE;
