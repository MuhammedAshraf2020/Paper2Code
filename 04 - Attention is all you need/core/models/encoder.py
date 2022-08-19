
from core.models.modules import *

class EncoderLayer(nn.Module):
    def __init__(self , d_model : int , heads : int  , dropout : float , device : str):
        super(EncoderLayer , self).__init__()

        self.attention        = MultiheadAttention(d_model , heads)
        self.attention_norm   = nn.LayerNorm(d_model)

        self.feedforward      = FeedForward(d_model , dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)

        self.dropout          = nn.Dropout(dropout)

    def forward(self , x , mask = None):
        x_att , _ = self.attention(x , x , mask)
        x = self.attention_norm(x + self.dropout(x_att))

        x_att = self.feedforward(x)
        x1 = self.feedforward_norm(x + self.dropout(x_att))

        return x1

class Encoder(nn.Module):
  def __init__(self , d_model : int , heads : int , N :int , encoder_vocab_size : int ,
                                               encoder_max_length : int ,  dropout : float , device : str):
    super(Encoder , self).__init__()

    self.device = device

    self.position_embedding = PositionEncoding(d_model = d_model , vocab_size =  encoder_vocab_size ,
                                                        max_length = encoder_max_length , device = device)

    self.encoder = nn.ModuleList([
        EncoderLayer(d_model= d_model , heads = heads , dropout = dropout , device = device) for _ in range(N)
    ])

    self.dropout = nn.Dropout(dropout)

  def forward(self , src , src_mask):
    src = self.dropout(self.position_embedding(src))
    for layer in self.encoder:
      src = layer(src , src_mask)

    return src
