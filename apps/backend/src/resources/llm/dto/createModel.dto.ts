import { IsEnum, IsInt, IsString } from 'class-validator';

export enum DatasetType {
  Parent = 'parentDataset',
  Child = 'childDataset',
}

export class CreateCsvFromParentDataset {
  @IsString()
  fileAlias: string;

  @IsInt()
  datasetId: number;

  @IsEnum(DatasetType)
  type: DatasetType = DatasetType.Parent;
}
