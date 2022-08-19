
import torch
import torch.nn as nn

def create_src_mask(src , src_pad_idx):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    # print(src_mask.shape)
    return src_mask

def create_trg_mask(trg , trg_pad_idx , device):
    # print("create mask " , trg.shape)
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    length = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((length , length) , device = device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    # print(trg_mask.shape)
    return trg_mask
