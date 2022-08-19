import torch
from .model_utils import *
import spacy

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        # print(batch)
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src.to("cuda")
            trg = batch.trg.to("cuda")

            output, _ = model(src, trg[:,:-1])
          
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

        
    return epoch_loss / len(iterator)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<sos>'] + tokens + ['<eos>']
        
    src_indexes = [src_field.stoi[token] for token in tokens]
    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = create_src_mask(src_tensor , src_pad_idx = src_field.stoi["<pad>"])
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.stoi["<sos>"]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = create_trg_mask(trg_tensor , trg_pad_idx=trg_field.stoi["<pad>"] , device = "cuda")
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.stoi["<eos>"]:
            break
    trg_tokens = [trg_field.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

def show_one_example( valid_data , SRC , TRG , transformer , DEVICE , example_idx = 30):
  src = vars(valid_data.examples[example_idx])['src']
  trg = vars(valid_data.examples[example_idx])['trg']
  translation, attention = translate_sentence(src, SRC, TRG, transformer , DEVICE)
  de_sen = ' '.join(src)
  en_sen = ' '.join(translation)
  return de_sen , en_sen