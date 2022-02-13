import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self , in_channels , out_channels , stride = 1 , subsample = False):
        super().__init__()

        self.block = nn.Sequential(
        # build resduial block that consist of (1x1 , 3x3 , 1x1) layers with respect
                            nn.Conv2d(in_channels   , out_channels       , kernel_size = 1 , stride = 1 , padding = 0) ,
                            nn.BatchNorm2d(out_channels)  ,
                            nn.LeakyReLU(0.02) ,

                            nn.Conv2d(out_channels  , out_channels       , kernel_size = 3 , stride = stride , padding = 1) ,
                            nn.BatchNorm2d(out_channels)  ,
                            nn.LeakyReLU(0.02) ,

                            nn.Conv2d(out_channels  , out_channels * 4   , kernel_size = 1 , stride = 1 , padding = 0) ,
                            nn.BatchNorm2d(out_channels * 4))

        self.downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels * 4 , kernel_size=1, stride=stride),
                            nn.BatchNorm2d(out_channels * 4)) if subsample else None

    def forward(self , x):
        z    = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out  = z + x
        return F.leaky_relu(out)


class Resnet(pl.LightningModule):
    def __init__(self , classes  , block_size , blocks = [64 , 128 , 256 , 512] ):
        super().__init__()

        self.classes = classes
        self.blocks = blocks
        self.block_size = block_size

        # create model layers in Sequential
        self.create_net = self.CreateNet()
        #initalize parameters in that model
        #self.init_params= self.InitParam()

    def CreateNet(self):

        # build stem layers at first
        self.stem = nn.Sequential(
                            nn.Conv2d(3, 32 , kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU(0.02) ,

                            nn.Conv2d(32 , 64 , kernel_size = 3 , padding = 1 , stride = 2 , bias = False),
                            nn.BatchNorm2d(64) ,
                            nn.LeakyReLU(0.02) )

        # stack layers of model togther
        learner = []
        current_channels = self.blocks[0]
        for idx , channels in enumerate(self.blocks):
            for time in range(self.block_size[idx]):
                # we need to downsample input of the block just in case of transmission from block to another
                stride = 2 if (time == 0 and idx > 0) else 1
                subsample = True if time == 0 else False
                # create resduial block
                resblock  = ResBlock(in_channels = current_channels , out_channels = channels , stride = stride , subsample = subsample)
                # add block to stack of blocks
                learner.append(resblock)
                current_channels = channels * 4

        self.learner = nn.Sequential(*learner)

        # create the classifier
        self.task  = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten() ,
                        nn.Linear(self.blocks[-1] * 4 , self.classes))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = nn.LeakyReLU)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self , x):
        x = self.stem(x)
        x = self.learner(x)
        x = self.task(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer ,
                                                   patience = 5 , verbose = True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self ,  batch , batch_idx):

        imgs , labels = batch
        preds = self(imgs)

        loss  = F.cross_entropy(preds , labels)
        acc   = (preds.argmax(dim = -1) == labels).float().mean()

        self.log("train_acc" , acc , on_step = False , on_epoch = True)
        self.log("train_loss" , loss)
        return loss

    def validation_step(self , batch , batch_idx):

        imgs , labels = batch
        preds = self(imgs)

        loss  = F.cross_entropy(preds , labels)
        acc = (preds.argmax(dim = -1) == labels).float().mean()

        self.log('val_acc' , acc)
        self.log("val_loss" , loss)


if __name__ == "__main__":
    model = Resnet(classes = 10 , block_size = [3 , 4 , 6 , 3] , blocks = [64 , 128 , 256 , 512] )
    data  = torch.randn((8 , 3 , 224 , 224 ))
    summary(model , (3 , 224 , 224))
