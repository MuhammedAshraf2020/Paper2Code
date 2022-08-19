
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import os

class DataModule:
    def __init__(self , batch_size , device):
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
            self.spacy_en = spacy.load('en_core_web_sm')
        except:
            os.system("python -m spacy download en_core_web_sm")
            os.system("python -m spacy download de_core_news_sm")
            self.spacy_de = spacy.load('de_core_news_sm')
            self.spacy_en = spacy.load('en_core_web_sm')

        self.BATCH_SIZE = batch_size
        self.DEVICE = device

    def create_src_trg_fields(self):
        SRC = Field(tokenize = self.tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

        TRG = Field(tokenize = self.tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

        return SRC , TRG

    def prepare(self):
        self.SRC , self.TRG =  self.create_src_trg_fields()
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (self.SRC, self.TRG))

        self.SRC.build_vocab(self.train_data, min_freq = 2)
        self.TRG.build_vocab(self.train_data, min_freq = 2)

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size = self.BATCH_SIZE,
            device = self.DEVICE)

    def tokenize_de(self , text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self , text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]


# SRC.build_vocab(train_data, min_freq = 2)
# TRG.build_vocab(train_data, min_freq = 2)

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#      batch_size = BATCH_SIZE,
#      device = DEVICE)
