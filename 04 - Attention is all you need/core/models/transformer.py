
from core.models.encoder import Encoder
from core.models.decoder import Decoder
from core.utils.model_utils import *

class Transformer(nn.Module):
    def __init__(self ,
                 device  : str , # represent device cpu / cuda
                 d_model : int , # embedding dim
                 max_length : int , # max num of words in one sentence
                 N : int ,  # number of repeated encoder and decoder
                 heads : int , # number of heads in multihead
                 encoder_vocab_size : int , # all words that we have in seq 1
                 decoder_vocab_size : int , # all words that we have in seq 2
                 dropout : float = 0.5 ,  # dropout rate
                 trg_pad_idx : int = 1 ,  # number that represent <pad> word in seq 1
                 src_pad_idx : int = 1 ): # number that represent <pad> word in seq 2

        super(Transformer , self).__init__()
        self.encoder = Encoder(d_model = d_model , heads = heads , N = N , encoder_vocab_size = encoder_vocab_size ,
                                                    encoder_max_length = max_length ,  dropout = dropout , device = device)

        self.decoder = Decoder( d_model = d_model , heads = heads  , N = N , decoder_vocab_size = decoder_vocab_size ,
                                                    decoder_max_length = max_length , dropout = dropout , device = device )

        self.trg_pad_idx = trg_pad_idx
        self.src_pad_idx = src_pad_idx

        self.device = device

        # self.device = device

    def forward(self , x , y ,  mask = None):

      # x = rearrange(x , "l b -> b l")
      # y = rearrange(y , "l b -> b l")

      # print("Im y im with sape equal to " , y.shape)

      src_mask = create_src_mask(x , self.src_pad_idx)
      trg_mask = create_trg_mask(y , self.trg_pad_idx , device = self.device)

      src =  self.encoder(x , src_mask)
      output , attention = self.decoder(y , src , trg_mask , src_mask)

      return output , attention
