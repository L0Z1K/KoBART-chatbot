import os
import argparse
import logging

from model import KoBART
from dataloader import ChatDataModule

import torch
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description="KoBART Chit-Chat")

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='train model')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--root_dir',
                    type=str,
                    default='logs',
                    help='path to save the model file')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    parser = KoBART.add_model_specific_args(parser)
    parser = ChatDataModule.add_model_specific_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = KoBART(args)
    dm = ChatDataModule(args)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.root_dir,
                                                       filename='{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=1,)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    if args.train:
        trainer.fit(model, dm)
    else:
        state_dict = torch.load(os.path.join(os.getcwd(), args.root_dir, 'last.ckpt'))["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v       # delete 'model.'
        
        model.model.load_state_dict(new_state_dict)

    if args.chat:
        model.model.eval()
        while 1:
            q = input('Q: ').strip()
            if q == 'quit':
                break
            print(f"A: {model.chat(q)}")