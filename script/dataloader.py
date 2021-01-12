import argparse

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

from transformers import PreTrainedTokenizerFast

class ChatDataset(Dataset):
    def __init__(self, filepath, tok_vocab, max_seq_len=128) -> None:
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)  # read csv format file.
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_vocab,
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
        ) # What is tok_vocab file? emji_tokenizer/model.json
    
    def __len__(self):
        return len(self.data)
    
    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens) # Convert token strings to id. (input should already TOKENIZE!)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:    # safe!
            while len(input_id) != self.max_seq_len: # while length becomes equal to max_len, do padding.
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0] # for ignoring padding token.
        else: # input's length is longer than max_len1 Then, just cut the input (Is this OK?)
            input_id = input_id[:self.max_seq_len-1] + [self.tokenizer.eos_token_id] # eos would be deleted by cutting. so adding again.
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask
    
    def __getitem__(self, index):
        # Dataset is organized by (Query, Answer), I think we don't need label with my training.
        record = self.data.iloc[index]
        q, a = record['Q'], record['A']
        q_tokens = [self.bos_token] + self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + self.tokenizer.tokenize(a) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(q_tokens, index) # I think we don't need index parameters!
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(a_tokens, index)

        labels = self.tokenizer.convert_tokens_to_ids(
            a_tokens[1:(self.max_seq_len+1)])   # train to predict next token? I don't know exactly.
        if len(labels) < self.max_seq_len:
            while len(labels) != self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100] # ?????????
        
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float),
                'labels': np.array(labels, dtype=np.int_)}

class ChatDataModule(pl.LightningDataModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
            
        parser.add_argument('--train_file',
                        type=str,
                        default='ChatbotData.csv',
                        help='train file')

        parser.add_argument('--tokenizer_path',
                            type=str,
                            default='emji_tokenizer/model.json',
                            help='tokenizer')

        parser.add_argument('--max_seq_len',
                            type=int,
                            default=32,
                            help='')

        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')

        return parser
        
    def setup(self, stage):
        data = ChatDataset(self.hparams.train_file,
                                 self.hparams.tokenizer_path,
                                 self.hparams.max_seq_len)

        train_size = int(len(data) * 0.9)
        test_size = len(data) - train_size
        self.train, self.test = random_split(data, [train_size, test_size])
    
    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.hparams.batch_size,
                           num_workers=self.hparams.num_workers,
                           shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers, shuffle=False)
        return val
        

        

