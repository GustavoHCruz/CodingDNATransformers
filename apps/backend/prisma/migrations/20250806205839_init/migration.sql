-- CreateEnum
CREATE TYPE "ProgressTypeEnum" AS ENUM ('COUNTER', 'PERCENTAGE');

-- CreateEnum
CREATE TYPE "ProgressStatusEnum" AS ENUM ('IN_PROGRESS', 'COMPLETE', 'FAILED');

-- CreateEnum
CREATE TYPE "ApproachEnum" AS ENUM ('EXINCLASSIFIER', 'TRIPLETCLASSIFIER', 'DNATRANSLATOR');

-- CreateEnum
CREATE TYPE "OriginEnum" AS ENUM ('GENBANK', 'GENCODE');

-- CreateEnum
CREATE TYPE "FeatureEnum" AS ENUM ('EXON', 'INTRON', 'CDS');

-- CreateTable
CREATE TABLE "ProgressTracker" (
    "id" SERIAL NOT NULL,
    "taskName" TEXT,
    "progress" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "progressType" "ProgressTypeEnum" NOT NULL DEFAULT 'COUNTER',
    "status" "ProgressStatusEnum" NOT NULL DEFAULT 'IN_PROGRESS',
    "info" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ProgressTracker_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ParentDataset" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "approach" "ApproachEnum" NOT NULL,
    "origin" "OriginEnum" NOT NULL,
    "recordCount" INTEGER DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ParentDataset_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ParentRecord" (
    "id" SERIAL NOT NULL,
    "sequence" TEXT NOT NULL,
    "target" TEXT NOT NULL,
    "flankBefore" TEXT,
    "flankAfter" TEXT,
    "organism" TEXT,
    "gene" TEXT,
    "parentDatasetId" INTEGER NOT NULL,

    CONSTRAINT "ParentRecord_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "RawFileInfo" (
    "id" SERIAL NOT NULL,
    "fileName" TEXT NOT NULL,
    "origin" "OriginEnum" NOT NULL,
    "totalRecords" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "RawFileInfo_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DNASequence" (
    "id" SERIAL NOT NULL,
    "sequence" TEXT NOT NULL,
    "accession" TEXT,
    "organism" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "DNASequence_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FeatureSequence" (
    "id" SERIAL NOT NULL,
    "sequence" TEXT NOT NULL,
    "type" "FeatureEnum" NOT NULL,
    "start" INTEGER NOT NULL,
    "end" INTEGER NOT NULL,
    "gene" TEXT,
    "strand" INTEGER,
    "before" TEXT,
    "after" TEXT,
    "dnaSequenceId" INTEGER NOT NULL,

    CONSTRAINT "FeatureSequence_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "GenerationBatch" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "GenerationBatch_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ChildDataset" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "approach" "ApproachEnum" NOT NULL,
    "recordCount" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "batchId" INTEGER NOT NULL,

    CONSTRAINT "ChildDataset_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ChildRecord" (
    "id" SERIAL NOT NULL,
    "childDatasetId" INTEGER NOT NULL,
    "parentRecordId" INTEGER NOT NULL,

    CONSTRAINT "ChildRecord_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ModelHistory" (
    "id" SERIAL NOT NULL,
    "modelAlias" TEXT NOT NULL,
    "checkpointName" TEXT NOT NULL,
    "approach" "ApproachEnum" NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "parentId" INTEGER,

    CONSTRAINT "ModelHistory_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TrainHistory" (
    "id" SERIAL NOT NULL,
    "learningRate" INTEGER NOT NULL,
    "batchSize" INTEGER NOT NULL,
    "gradientAccumulation" INTEGER NOT NULL,
    "warmupRatio" DOUBLE PRECISION NOT NULL,
    "epochs" INTEGER NOT NULL,
    "hideProb" DOUBLE PRECISION,
    "lora" BOOLEAN NOT NULL DEFAULT false,
    "loss" DOUBLE PRECISION NOT NULL,
    "durationSec" INTEGER NOT NULL,
    "seed" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "modelId" INTEGER NOT NULL,
    "trainDatasetId" INTEGER,
    "evalDatasetId" INTEGER,

    CONSTRAINT "TrainHistory_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "EvalHistory" (
    "id" SERIAL NOT NULL,
    "loss" DOUBLE PRECISION NOT NULL,
    "accuracy" DOUBLE PRECISION NOT NULL,
    "durationSec" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "modelId" INTEGER NOT NULL,

    CONSTRAINT "EvalHistory_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "ParentRecord" ADD CONSTRAINT "ParentRecord_parentDatasetId_fkey" FOREIGN KEY ("parentDatasetId") REFERENCES "ParentDataset"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "FeatureSequence" ADD CONSTRAINT "FeatureSequence_dnaSequenceId_fkey" FOREIGN KEY ("dnaSequenceId") REFERENCES "DNASequence"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ChildDataset" ADD CONSTRAINT "ChildDataset_batchId_fkey" FOREIGN KEY ("batchId") REFERENCES "GenerationBatch"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ChildRecord" ADD CONSTRAINT "ChildRecord_childDatasetId_fkey" FOREIGN KEY ("childDatasetId") REFERENCES "ChildDataset"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ChildRecord" ADD CONSTRAINT "ChildRecord_parentRecordId_fkey" FOREIGN KEY ("parentRecordId") REFERENCES "ParentRecord"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ModelHistory" ADD CONSTRAINT "ModelHistory_parentId_fkey" FOREIGN KEY ("parentId") REFERENCES "ModelHistory"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TrainHistory" ADD CONSTRAINT "TrainHistory_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "ModelHistory"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TrainHistory" ADD CONSTRAINT "TrainHistory_trainDatasetId_fkey" FOREIGN KEY ("trainDatasetId") REFERENCES "ChildDataset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TrainHistory" ADD CONSTRAINT "TrainHistory_evalDatasetId_fkey" FOREIGN KEY ("evalDatasetId") REFERENCES "ChildDataset"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "EvalHistory" ADD CONSTRAINT "EvalHistory_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "ModelHistory"("id") ON DELETE CASCADE ON UPDATE CASCADE;
