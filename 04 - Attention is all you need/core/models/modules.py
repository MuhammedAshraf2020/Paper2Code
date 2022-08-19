
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat , rearrange

class PositionEncoding(nn.Module):
    def __init__(self , d_model : int , vocab_size : int ,  max_length : int , device : str , dropout = 0.1):
        super().__init__()

        self.device = device
        # define embedding layer
        self.EmbedLayer    = nn.Embedding(vocab_size , d_model)
        self.PositionLayer = nn.Embedding(max_length , d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self , x):
        # print("embed = " , x.device)
        b , l = x.shape # batch_size , length
        # print("Im position encoding im with shape" , x.shape)
        # embedding the layer
        x   = self.EmbedLayer(x)
        # get positional embedding for every word that we have
        pos = repeat(torch.arange(0 , l) , "... -> b ..." , b = b).to(self.device)
        embed_pos = self.PositionLayer(pos)
        return self.dropout(x + embed_pos)


class FeedForward(nn.Module):
    def __init__(self , d_model : int , dropout : float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.model = nn.Sequential(
                    nn.Linear(d_model , 512) ,
                    nn.ReLU() ,
                    nn.Linear(512 , d_model))

        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self , x):
        a = self.model(x)
        return self.norm_layer(self.dropout(a) + x)


class MultiheadAttention(nn.Module):
    def __init__(self , d_model : int , heads : int ):
        super().__init__()
        self.heads = heads
        # validate that the number of embed dim is divideable on heads
        if d_model % heads != 0:
            raise ValueError(f"Embed dim {d_model} should be divideable on heads {heads}")

        self.query_size = d_model // heads
        self.scale = 1.0 / (self.query_size ** 0.5)
        # define key , val , Q matrix
        self.key_mat   = nn.Linear(self.query_size , self.query_size)
        self.query_mat = nn.Linear(self.query_size , self.query_size)
        self.value_mat = nn.Linear(self.query_size , self.query_size)

        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self , kv , q , mask = None):
        #reshape x of shape (b , l , d) -> (b , h , l , q)

        kv = rearrange(kv , "b l (h d) -> b h l d" , h = self.heads , d = self.query_size)
        _q  = rearrange(q  , "b l (h d) -> b h l d" , h = self.heads , d = self.query_size)
        # apply some Linear transformation to Q , K , V

        keys    = self.key_mat(kv)
        queries = self.query_mat(_q)
        values  = self.value_mat(kv)
        # multiply Q AND K.T

        qkT = torch.einsum("bhqd , blkd -> blqk" , queries , keys)
        scores = qkT * self.scale
        #apply mask if exist
        # print("score shape is " , scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0 , -1e20)
        # scale output and normalize it by apply softmax
        scores = F.softmax(scores , dim = -1)
        # get attention of K , V , Q -> softmax((Q.KT) / sqrt(q_dim)) * V
        attention = torch.einsum("bhkd , bhqk -> bhqd" , values , scores)

        # Concat all heads togther
        output = rearrange(attention , "b h l d -> b l (h d)")
        return output , scores
