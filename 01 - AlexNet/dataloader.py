
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


class IntelData(pl.LightningDataModule):
  def __init__(self , train_path , valid_path , batch_size = 32):
    super().__init__()
    
    self.train_path = train_path
    self.valid_path = valid_path
    
    self.batch_size = batch_size
    
    self.transform = transforms.Compose([transforms.ToTensor() , 
                           transforms.Resize((227 , 227)) ,
												   #transforms.RandomCrop(227 , 227) , 
												   #transforms.RandomHorizontalFlip(p=0.5) ,
												   transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])
    
  def prepare(self):
    self.train = torchvision.datasets.ImageFolder(self.train_path , transform = self.transform)
    self.valid = torchvision.datasets.ImageFolder(self.valid_path , transform = self.transform)
    
  def train_dataloader(self):
    return DataLoader(self.train , shuffle = True , batch_size = self.batch_size)
    
  def val_dataloader(self):
    return DataLoader(self.valid , shuffle = True , batch_size = self.batch_size)