
import time
from options.train_options import TrainOptions
from data import CustomDataModule
from models import create_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from util.util import  SetupCallback
from pytorch_lightning import loggers as pl_loggers
if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
      # create a model given opt.model and other options

    if opt.seed is not None:
        pl.seed_everything(opt.seed)
    
    if opt.accelerator == 'ddp':
        opt.batch_size = int(opt.batch_size / max(1, opt.gpus))
        opt.num_threads = int(opt.num_threads / max(1, opt.gpus))
        opt.sync_batchnorm = True
  
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='best-{epoch:02d}-{val_error:.4f}', monitor='val_error', save_last=True, mode='min')

    epoch_checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch:02d}-{val_error:.4f}', save_top_k=-1, monitor=None)

    setup_callback = SetupCallback()

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    callbacks = [best_checkpoint_callback, epoch_checkpoint_callback, lr_monitor]


    data = CustomDataModule(opt)
    model = create_model(opt)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = opt.default_root_dir, name="lightning_logs")
    trainer = Trainer.from_argparse_args(opt, callbacks=callbacks, logger=tb_logger)
    # trainer.logger.default_hp_metric = False
    #auto_scale_batch_size = True
    if not opt.valid:
        trainer.fit(model, data)
    else:
        model = model.load_from_checkpoint("/home1/caixin/FewShotGaze/logs/gaze_estimation_xgaze/lightning_logs/version_70/checkpoints/best-epoch=28-val_error=5.1344.ckpt", write_features = opt.write_features, cam = opt.cam, visual_freq = 400)
        trainer.validate(model, data)
