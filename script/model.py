import argparse

import pytorch_lightning as pl
import torch

from kobart import get_pytorch_kobart_model
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

class KoBART(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.hparams = hparams

        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model()) # load pretrained kobart model
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.hparams.tokenizer_path,
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        return parser

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'], # Indices of input sequence tokens  
                          attention_mask=inputs['attention_mask'], # when two input's length is different, we should pad one of them with 0 values(example). But model must avoid performing attention on padding token. So masked values help model to avoid padding token.
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'], # same reason with attention_mask
                          labels=inputs['labels'],
                          return_dict=True) # return index tuple instead of plain text.

    def training_step(self, batch, batch_idx):
        outs = self(batch) # outs : return value of BartForConditionalGeneration
        loss = outs.loss # loss will exist when you set the label.
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = [loss for loss in outputs]
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
        
    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters()) # model.parameters() + parameter's name (ex, model.decoder.layernorm_embedding.bias)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # should check this elements that really don't use weight decay.
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(   # parameters that have weight_decay.
                nd in n for nd in no_decay)], 'weight_decay': 0.01},    # check by their name.
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)

        data_len = len(self.train_dataloader().dataset)
        num_train_steps = int(data_len / self.hparams.batch_size * self.hparams.max_epochs) # total number of train steps
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(    # Set the learning rate Scheduler
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )

        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]
    
    def chat(self, text):
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        res_ids = self.model.generate(input_ids=torch.tensor([input_ids]),
                                      max_length=self.hparams.max_seq_len,
                                      num_beams=5,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      bad_words_ids=[[self.tokenizer.unk_token_id]])
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace('<s>', '').replace('</s>', '').strip()