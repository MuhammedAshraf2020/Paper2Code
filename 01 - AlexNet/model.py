from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

class AlexNet(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
				
        nn.Conv2d(in_channels = 3  , out_channels = 96  , kernel_size = 11 , stride = 4 ) , 
				nn.ReLU() ,
				nn.MaxPool2d(kernel_size = 3 , stride = 2) ,

				nn.Conv2d(in_channels = 96 , out_channels = 256 , kernel_size = 5  , stride = 1  , padding = 2) ,
				nn.ReLU() ,
				nn.MaxPool2d(kernel_size = 3 , stride = 2) ,
				
        nn.Conv2d(in_channels = 256, out_channels = 384 , kernel_size = 3  , stride = 1  , padding = 1) ,
				nn.ReLU() ,
				
        nn.Conv2d(in_channels = 384, out_channels = 384 , kernel_size = 3  , stride = 1  , padding = 1) ,
				nn.ReLU() ,
				
        nn.Conv2d(in_channels = 384, out_channels = 256 , kernel_size = 3  , stride = 1  , padding = 1) ,
				nn.ReLU() ,
				nn.MaxPool2d(kernel_size = 3 , stride = 2) ,
        nn.Flatten() ,
				nn.Linear(in_features = 9216 , out_features = 4096 // 2) ,
				nn.ReLU() ,
        nn.Dropout(0.5) ,
				# nn.Linear(in_features = 4096 , out_features = 4096) ,
				# nn.ReLU() ,
				# nn.Dropout(0.5) ,
        nn.Linear(in_features = 4096 // 2 , out_features = 8) )
    
  def forward(self , img):
    x = self.model(img)
    return x
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters() , lr = 1e-3)		
    return optimizer
  
  def training_step(self , train_batch , batch_idx):
    x , y = train_batch
    outputs = self(x)
    loss = F.cross_entropy(outputs , y)
    preds = torch.argmax(outputs , dim = 1)
    
    train_acc = accuracy_score(preds.cpu() , y.cpu())
    train_acc = torch.tensor(train_acc)

    self.log("train_loss", loss , prog_bar = True )
    self.log("train acc" , train_acc , prog_bar = True)
    return loss
  
  def validation_step(self , val_batch , batch_idx):
    x , y = val_batch
    outputs = self(x)
    
    loss = F.cross_entropy(outputs , y)
    preds = torch.argmax(outputs , dim = 1)

    val_acc = accuracy_score(preds.cpu() , y.cpu())
    val_acc = torch.tensor(val_acc)

    self.log("val_loss", loss , prog_bar = True)
    self.log("val_acc" , val_acc , prog_bar = True)


