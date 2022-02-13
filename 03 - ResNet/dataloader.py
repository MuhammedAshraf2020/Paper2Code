import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class dataloader(pl.LightningDataModule):
    def __init__(self ,  train_path , valid_path , batch_size = 32):
        super().__init__()

        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path

        self.transforms = transforms.Compose([

                transforms.ToTensor() ,
                transforms.Resize((224 , 224)) ,
                transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
        ])

    def prepare(self):
        self.train_data = torchvision.datasets.ImageFolder(self.train_path , transform = self.transforms)
        self.valid_data = torchvision.datasets.ImageFolder(self.valid_path , transform = self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_data , shuffle = True , batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data , shuffle = True , batch_size = self.batch_size)
