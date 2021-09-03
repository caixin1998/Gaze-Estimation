
import time
from options.train_options import TrainOptions
from data import CustomDataModule
from models import create_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from util.util import  SetupCallback
if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
      # create a model given opt.model and other options

    if opt.seed is not None:
        pl.seed_everything(opt.seed)
    
    if opt.accelerator == 'ddp':
        opt.batch_size = int(opt.batch_size / max(1, opt.gpus))
        opt.workers = int(opt.num_threads / max(1, opt.gpus))
  
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='best-{epoch:02d}-{val_error:.4f}', monitor='val_error', save_last=True, mode='min')

    epoch_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch:02d}-{val_error:.4f}', save_top_k=-1, monitor=None)

    setup_callback = SetupCallback()

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    callbacks = [best_checkpoint_callback, epoch_checkpoint_callback, lr_monitor]


    data = CustomDataModule(opt)
    model = create_model(opt)
    trainer = Trainer.from_argparse_args(opt, callbacks=callbacks)

    trainer.fit(model, data)

