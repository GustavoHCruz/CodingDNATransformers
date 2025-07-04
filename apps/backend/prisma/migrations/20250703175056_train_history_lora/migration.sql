-- AlterTable
ALTER TABLE "TrainHistory" ADD COLUMN     "lora" BOOLEAN NOT NULL DEFAULT false,
ALTER COLUMN "hideProb" DROP NOT NULL;
