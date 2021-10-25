import configargparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, filename=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.filename = filename
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path', default = self.filename)
        parser.add_argument('--name', type=str, default='gaze_estimation', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--seed', type=int, default=None, help='random seed for experiments.')

        parser.add_argument('--accelerator', type=str, default="ddp", help='accelerator  for experiments.')

        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        # help='evaluate model on specific set')
        parser.add_argument('--checkpoints_dir', type=str, default='./logs', help='models are saved here')
        # model parameters
        parser.add_argument('--criterion', type=str, default='smoothl1', help='models are saved here')

        # parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        # parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        # parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        # parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        # parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        # parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        # parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        
        #for model
        parser.add_argument('--model', type=str, default='gaze', help='chooses which model to use. ')

        parser.add_argument('--netGaze', type=str, default='regressor', help='type for network')
        parser.add_argument('--backbone', type=str, default='resnet50', help='backbone for network')
        parser.add_argument('--ngf', type=int, default='128', help='network filters in the last conv layer')
        parser.add_argument('--metric', type=str, default='angular', help='metrics')
        # parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
     

        #for dataloader
        parser.add_argument('--dataset', type=str, default='xgaze', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        parser.add_argument('--num_threads', default=16, type=int, help='# threads for loading data')


        # for dataset
        parser.add_argument('--dataroot',  type=str, default= "/home/caixin/GazeData/xgaze_224", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        
        parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        parser.add_argument('--flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--preprocess', type =str, default = "none", help='new dataset option')

        #for trainer
        parser.add_argument('--gpus', type=int, default=4, help='number of gpus')
        parser.add_argument('--resume_from_checkpoint', type= str, default = None, help='resume_from_checkpoint and recover the whole training for Trainer.')
        parser.add_argument('--valid', action='store_true', help='use fit or validate')


        #for cam visualizer
        parser.add_argument('--cam', action='store_true', help='if specified, use cam to visualize the conv attention.')
   

        parser.set_defaults(
        max_epochs=20,
        check_val_every_n_epoch=1,
        weights_summary='full',
        log_every_n_steps=20,
        terminate_on_nan = True,
        gradient_clip_val=1.0,
#         track_grad_norm=2,
        )

        self.initialized = True
        return parser





    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = configargparse.ArgumentParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        opt.default_root_dir = expr_dir
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.name = opt.name + "_" + opt.dataset
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)


        self.opt = opt
        return self.opt
