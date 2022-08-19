import os
import hydra
import time
import numpy as np
import pickle
import wandb 

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from core.models import Transformer
from core.dataset.data import DataModule
from core.utils.train_utils import train , evaluate , show_one_example
#TODO save save_checkpoint

@hydra.main(config_path = "./configs" , config_name = "config")
def main(cfg):
    np.random.seed(cfg.training.SEED)
    torch.cuda.manual_seed(cfg.training.SEED)
    torch.backends.cudnn.deterministic = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    data = DataModule(batch_size = cfg.training.BATCH_SIZE , device = DEVICE)
    data.prepare()
    SRC , TRG = data.SRC , data.TRG
    valid_data = data.valid_data
    
    if not os.path.exists("/content/tokenizers"):
      os.mkdir("/content/tokenizers")

    # Save SRC / TRG Tokenizers
    dumps_data = {"SRC" : SRC.vocab , 
                  "TRG" : TRG.vocab}

    with open("/content/tokenizers/tokenizer.pickle" , "wb") as handle:
      pickle.dump(dumps_data , handle , pickle.HIGHEST_PROTOCOL)    

    src_length = len(SRC.vocab.stoi)
    trg_length = len(TRG.vocab.stoi)

    wandb.init(
      # Set the project where this run will be logged
      project="attention-is-all-you-need", 
      entity = "muhammed266" , 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": cfg.training.LR,
      "architecture": "Transformer",
      "dataset": "Multi30K",
      "epochs": 30,
      })    
    # Lets define Transformer model with our configurations
    transformer = Transformer(d_model = cfg.model.D_MODEL , max_length = cfg.model.MAX_WORDS , N = cfg.model.N ,
                             heads = cfg.model.HEADS , encoder_vocab_size  = src_length ,
                             decoder_vocab_size = trg_length , dropout = cfg.model.DROPOUT_RATE , device = DEVICE).to(DEVICE)

    opt = Adam(transformer.parameters() , lr = cfg.training.LR)

    loss_fn = CrossEntropyLoss(ignore_index = SRC.vocab.stoi["<pad>"])
    checkpoint_path = os.path.join(os.getcwd() , cfg.training.CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    best_loss = 100
    CLIP = 1  
    for epoch in range(cfg.training.N_EPOCHS):

        # train and evaluate our model
        train_loss = train(transformer , data.train_iterator , opt , loss_fn , CLIP)
        valid_loss = evaluate(transformer , data.valid_iterator , loss_fn)

        if (epoch + 1) % 2 == 0:
            # lets display some translated sentences from our model
            sentence , out  = show_one_example(valid_data , SRC.vocab , TRG.vocab , transformer , DEVICE , example_idx = torch.randint(1 , 200 , (1,)))
            print("\ngerman sentence is : " , sentence)
            print("our model output is : " , out , end = "\n\n")
        
        wandb.log({"train loss" : train_loss ,
                          "val loss" : valid_loss })
        # lets save the model that we have in while the training process
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_path = os.path.join(checkpoint_path , f"transformer-epoch({epoch}).pt")
            torch.save(transformer.state_dict() , save_path)
            print(f"Saved new checkpoint with loss = {valid_loss:.3f} in {save_path}")

        print(f'Epoch: {epoch+1:02}')
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')

if __name__== "__main__":
  main()
