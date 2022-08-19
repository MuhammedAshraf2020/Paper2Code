
from core.models.modules import *

class Decoder(nn.Module):
  def __init__(self , d_model , heads  , N , decoder_vocab_size , decoder_max_length , dropout , device ):
    super(Decoder , self).__init__()
    self.device = device

    self.position_embedding = PositionEncoding(d_model = d_model , vocab_size = decoder_vocab_size,
                                                max_length = decoder_max_length , dropout = dropout , device = device)
    self.decoder = nn.ModuleList([
        DecoderLayer(d_model = d_model , heads = heads , dropout = dropout , device = device)
    ])

    self.fc      = nn.Linear(d_model , decoder_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self , trg , src , trg_mask , src_mask):
    trg = self.position_embedding(trg)

    for layer in self.decoder:
      trg , attention = layer(trg , src , trg_mask , src_mask)

    output = self.fc(trg)
    return output , attention

class DecoderLayer(nn.Module):
  def __init__(self , d_model : int , heads : int , dropout : float , device : str):
    super(DecoderLayer , self).__init__()

    self.attention        = MultiheadAttention(d_model , heads)
    self.masked_attention = MultiheadAttention(d_model , heads)

    self.feedforward      = FeedForward(d_model , dropout)

    self.masked_norm      = nn.LayerNorm(d_model)
    self.attention_norm   = nn.LayerNorm(d_model)
    self.feedforward_norm = nn.LayerNorm(d_model)

    self.dropout          = nn.Dropout(dropout)

  def forward(self , trg , src , trg_mask , src_mask):
    # print("im in decoder im with shape " , trg.shape)

    trg_att , _ = self.masked_attention(trg , trg , trg_mask)
    trg = self.masked_norm(trg + self.dropout(trg_att))

    # print("after masked attention im with shape" , trg.shape)
    trg_att , attention = self.attention(src , trg , src_mask)
    trg = self.attention_norm(trg +  self.dropout(trg_att))

    trg_att = self.feedforward(trg)
    trg = self.feedforward_norm(trg + self.dropout(trg_att))

    return trg , attention





# if __name__ == "__main__":
#   sentence = torch.randint(1 , 1999 , (32 , 20))
#   transformer = Transformer(d_model = 50 , max_length = 20 , N = 8 , heads = 10 ,
#                              encoder_vocab_size  =  2000 , decoder_vocab_size = 3000 , dropout =  0.5 , device = "cpu").to("cpu")

# #     # print("multihead attention shape =" , multihead(data , data).shape)
# #     # #
# #     # mlp = FeedForward(50)
# #     # print("mlp output shape =" , mlp(data).shape)

# #     # encoder = Encoder(d_model = 50 , heads = 10 , dropout = 0.5)
# #     # print("encoder output shpe =" ,encoder(data).shape)

#   # src_mask = create_src_mask(sentence , 0)
#   # trg_mask = create_trg_mask(sentence , 0 , "cuda")

# #     # decoder = Decoder(d_model = 50 , heads = 10 , dropout = 0.5)
# #     # print("decoder output shape =" ,decoder(data , data , trg_mask , src_mask).shape)
