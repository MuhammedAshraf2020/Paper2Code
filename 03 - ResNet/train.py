import hydra
import pytorch_lightning as pl
from torchsummary import summary
from dataloader import dataloader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import Resnet


def ResNet(classes : int , model_version : str = "ResNet50"):
    if model_version == "ResNet50":
        archit = [3 , 4 , 6 , 3]
    elif model_version == "ResNet101":
        archit = [3 , 4 , 23 , 3]
    elif model_version == "ResNet150":
        archit = [3 , 8 , 36 , 3]
    else:
        raise ValueError("This model version {model_version} is not exist".format(model_version = model_version))

    return Resnet(classes = classes , block_size = archit)


@hydra.main(config_path = "./configs" , config_name = "config")
def main(cfg):
    Data   = dataloader(cfg.dataset.train_data_path , cfg.dataset.valid_data_path , batch_size = cfg.training.batch_size )
    Model  = ResNet(cfg.model.classes , model_version = cfg.model.model_version)
    #summary(Model , input_size = (3 , 224 , 224))
    Data.prepare()

    checkpoint_callback = ModelCheckpoint(cfg.training.save_checkpoint_path  , monitor = "val_loss" , mode = "min")
    early_stopping = EarlyStopping(monitor = "val_loss" , patience = cfg.training.early_stopping_patience , verbose = True , mode = "min")
    wandb_logger = WandbLogger(project = "resnet" , entity = "muhammed266")

    trainer = pl.Trainer(log_every_n_steps = cfg.training.log_every_n_steps , deterministic = cfg.training.deterministic , gpus = 1 , max_epochs = cfg.training.max_epochs , logger = wandb_logger ,
                            fast_dev_run = False , callbacks = [checkpoint_callback , early_stopping])
    trainer.fit(Module , Data)

if __name__ == "__main__":
    main()
