from core.utils.train_utils import translate_sentence
from core.models import Transformer
from hydra import compose, initialize
import matplotlib.ticker as ticker

import spacy 
import matplotlib.pyplot as plt
import argparse
import torch
import pickle
import hydra
import io
from PIL import Image
import matplotlib.pyplot as plt

def load_src_trg_tokenizers(tokenizers_path):
    with open(tokenizers_path , 'rb') as handle:
        tokenizers = pickle.load(handle)
    src_tokenizer , trg_tokenizer = tokenizers["SRC"] , tokenizers["TRG"]
    return src_tokenizer , trg_tokenizer

def display_attention(sentence, translation, attention):
    sentence = sentence.split()

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,1, 1)
    _attention = attention.squeeze(0)[0].cpu().detach().numpy()
    cax = ax.matshow(_attention, cmap='bone')
    ax.tick_params(labelsize=32)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                           rotation=45)
    ax.set_yticklabels(['']+translation)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.savefig("attention.png")
    plt.close()

def main(cfg , sentence , DEVICE , model_path , tokenizers_path , visualize_attention):
    src , trg = load_src_trg_tokenizers(tokenizers_path)
    model = Transformer(d_model = cfg.model.D_MODEL , max_length = cfg.model.MAX_WORDS , N = cfg.model.N ,
                             heads = cfg.model.HEADS , encoder_vocab_size  = len(src.stoi) ,
                             decoder_vocab_size = len(trg.stoi) , dropout = cfg.model.DROPOUT_RATE , device = DEVICE).to(DEVICE)

    model.load_state_dict(torch.load(model_path))
    translation , attention = translate_sentence(sentence ,  src , trg , model , DEVICE )
    print(translation)
    if visualize_attention:
      display_attention(sentence , translation , attention)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('-sentence' , '--sentence' , default = 'ein mann rollt einen runden tisch über den boden .' ,
                                                                    required=True ,  help='input any german sentence here')

    parser.add_argument('-model_path' , '--model_path'  ,default = '/content/checkpoint/transformer-epoch(25).pt' ,
                                                                    required=True, help='Path to the model (Transformer)')

    parser.add_argument('-tokenizers_path' , '--tokenizers_path' ,default = '/content/tokenizers/tokenizer.pickle',
                                                                   required = True , help='Path to the src / trg - tokenizers ')
    
    parser.add_argument('-visualize_attention' , '--visualize_attention' , action='store_true'
                                                                   , help='Path to the src / trg - tokenizers ')
    
    args = parser.parse_args()
    with initialize(config_path="./configs", job_name="transformer"):
        cfg = compose(config_name="config")
        main(cfg , args.sentence , device , args.model_path , args.tokenizers_path , args.visualize_attention)
