from model import AlexNet
import pytorch_lightning as pl
from dataloader import IntelData
from wandb_loggers import logVisualizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(train_path , val_path , save_path ,batch_size):
  Data  = IntelData(train_path , val_path , batch_size )
  Model = AlexNet()
  Data.prepare()
  
  checkpoint_callback = ModelCheckpoint(save_path , monitor = "val_loss" , mode = "min")
  early_stopping = EarlyStopping(monitor = "val_loss" , patience = 5 , verbose = True , mode = "min")
  
  wandb_logger = WandbLogger(project="AlexNet", entity="muhammed266")
  trainer = pl.Trainer(log_every_n_steps = 20 , deterministic=True, gpus = 1 , max_epochs = 20 , logger = wandb_logger ,
								fast_dev_run = False , callbacks = [checkpoint_callback   , early_stopping])
                
  trainer.fit(Model , Data)


if __name__ == "__main__":
	train_path = "/content/seg_train/seg_train"
	valid_path = "/content/seg_test/seg_test"
	main(train_path , valid_path , "weights" , 32 )