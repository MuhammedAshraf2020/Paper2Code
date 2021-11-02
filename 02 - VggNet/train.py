
import argparse
from models import Model as vgg
import pytorch_lightning as pl
from torchsummary import summary
from data_loader import dataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(train_path , val_path , save_path , batch_size , model_num):
  Data  = dataLoader(train_path , val_path , batch_size )
  Model = vgg(modelType = model_num)
  print(summary(Model.to("cuda") , input_size = (3 , 224 , 224)))
  Data.prepare()
  
  checkpoint_callback = ModelCheckpoint(save_path , monitor = "val_loss" , mode = "min")
  early_stopping = EarlyStopping(monitor = "val_loss" , patience = 5 , verbose = True , mode = "min")
  
  wandb_logger = WandbLogger(project="vgg", entity="muhammed266")
  trainer = pl.Trainer(log_every_n_steps = 20 , deterministic=True, gpus = 1 , max_epochs = 30 , logger = wandb_logger ,
								fast_dev_run = False , callbacks = [checkpoint_callback   , early_stopping])
                
  trainer.fit(Model , Data)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_num"  , help = "Model type one of (A , B , C , D , E) values")
	args = parser.parse_args()

	model_num = args.model_num

	train_path = "/content/seg_train/seg_train"
	valid_path = "/content/seg_test/seg_test"
	main(train_path , valid_path , "weights" , 32 , model_num )