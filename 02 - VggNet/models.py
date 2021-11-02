
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import accuracy_score



class ConvLayer(nn.Module):
	def __init__(self , previous_channels , kernel_size , currnet_channels , stride , padding ):
		super().__init__()

		self.model = nn.Sequential(
				nn.Conv2d(previous_channels , currnet_channels , kernel_size , stride = stride , padding = padding) ,
				nn.ReLU()
			)

	def forward(self , x):
		return self.model(x)


class Model(pl.LightningModule):
  def __init__(self , modelType = None , classes = 6):
    super().__init__()
    
    self.classes = classes
		# Check if type is valid or not
    if type(modelType) != str :
      raise ValueError("Type of modelType must by string") 

		#choice between types of existing Models
    if modelType == "A":
      self.archit = architA
    elif modelType == "B":
      self.archit = architB
    elif modelType == "C":
      self.archit = architC
    elif modelType == "D":
      self.archit = architD
    elif modelType == "E":
      self.archit = architE
    else:
      raise ValueError("Your model must be one of these models (A , B , C , D , E)")

		#Build Model which have been choicen before
    featureExtractor = self.buildBody()
    fullyConnected   = self.buildTail()
    self.model = nn.Sequential(featureExtractor , fullyConnected )
    
  def forward(self , x):
    return self.model(x)
    
  def buildBody(self):
    layers = []
    previous_channels = 3
		# walk through every layer in our archit
    for layer in self.archit:
			# if layer is string > maxpooling 
      if type(layer) == str:
        layers += [nn.MaxPool2d(kernel_size = 2 , stride = 2)]
			# else walk through number of repeated layer 
      else:
        currnet_channels = layer[0]
				# add repeated layers to our sequential model
        for repeat in range(layer[-1]):
          layers += [ConvLayer(previous_channels = previous_channels  , currnet_channels = currnet_channels , kernel_size = layer[1] , 
                                                            stride = layer[2] , padding = layer[3])]
					#update channels to the next layer
          previous_channels = currnet_channels
		# collect all of them togther in one sequential model
    return nn.Sequential(*layers)
    
  def buildTail(self):

    layers = nn.Sequential(
					nn.Flatten() ,
	
					nn.Linear(in_features = 7 * 7 * 512 , out_features = 4096 // 2) ,
					nn.ReLU() ,
					nn.Dropout(0.5) ,
					
					nn.Linear(in_features = 4096 // 2 , out_features = self.classes) 
			)
    return layers
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr = 0.001)
    return {
					"optimizer" : optimizer ,
					
					"lr_scheduler": {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3 , patience = 2 ),
            "monitor": "val_loss"}
		}
  
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



"""
Conjection which i will follow 
every layer [filters , kernal , stride , padding , repeate_num]
"""
# The First architecture 11 layer vgg 11
architA = [[64  , 3 , 1 , 1  , 1] ,
		   "M"  ,
		  [128  , 3 , 1 , 1  , 1] ,
		   "M"  ,
		  [256  , 3 , 1 , 1  , 2] , 
		   "M"  , 
		  [512  , 3 , 1 , 1  , 2] , 
		   "M"  ,
		  [512  , 3 , 1 , 1  , 2] ,
		   "M"]

# The Second architecture 13 Layer vgg 13
architB = [[64  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [128  , 3 , 1 , 1  , 2] ,
		   "M"  ,
 		  [256  , 3 , 1 , 1  , 2] , 
 		   "M"  ,
		  [512  , 3 , 1 , 1  , 2] , 
		   "M"  ,
		  [512  , 3 , 1 , 1  , 2] ,
	  	   "M"]

#The Third architecture 16 layer vgg 16 (1 conv)
architC = [[64  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [128  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [256  , 3 , 1 , 1  , 2] ,
		  [256  , 1 , 1 , 0  , 1] , 
		   "M"  ,
		  [512  , 3 , 1 , 1  , 2] ,
		  [512  , 1 , 1 , 0  , 1] , 
		   "M"  ,
		  [512  , 3 , 1 , 1  , 2] ,
		  [512  , 1 , 1 , 0  , 1] ,
		   "M"  , ]

#The Fourth architecture 16 layer vgg 16 (3 conv)
architD = [[64  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [128  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [256  , 3 , 1 , 1  , 3] ,
		   "M"  ,
		  [512  , 3 , 1 , 1  , 3] ,
		   "M"  ,
		  [512  , 3 , 1 , 1  , 3] ,
		   "M"  , ]

#The Fifth architecture 16 layer vgg 16 (3 conv)
architE = [[64  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [128  , 3 , 1 , 1  , 2] ,
		   "M"  , 
		  [256  , 3 , 1 , 1  , 4] ,
		   "M"  ,
		  [512  , 3 , 1 , 1  , 4] ,
		   "M"  ,
		  [512  , 3 , 1 , 1  , 4] ,
		   "M"  , ]

