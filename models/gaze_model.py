"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
from pytorch_lightning import LightningModule
import torchvision
from metrics import MeanDistanceError,MeanAngularError
from util.data_util import draw_point
class GazeModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--use_eyes', action='store_true', default=False, help='use eyes for itracker model.')

        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

        return parser

    def __init__(self, **kwargs):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, **kwargs)
        
        opt = self.opt
        
        
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netGazeNetwork = networks.define_GazeNetwork(opt.netGaze, opt.backbone, opt.ngf)

        self.criterionLoss = torch.nn.SmoothL1Loss()

        if opt.metric == "distance":
            self.train_metric = MeanDistanceError(dist_sync_on_step=True)
            self.valid_metric = MeanDistanceError(dist_sync_on_step=True)
        elif opt.metric == "angular":
            self.train_metric = MeanAngularError(dist_sync_on_step=True)
            self.valid_metric = MeanAngularError(dist_sync_on_step=True)

        self.model_names = ["GazeNetwork"]
    def set_input(self, input):
        self.net_input = {}
        if "face" in input:
            self.face = input["face"]
            self.net_input["face"] = self.face
        if "leye" in input:
            self.leye = input['leye']
            self.net_input["leye"] = self.leye
        if "reye" in input:
            self.reye = input['reye']
            self.net_input["reye"] = self.reye
        if "ec" in input:
            self.ec = input['ec']
            self.net_input["ec"] = self.ec
        if "gt_position" in input:
            self.gt_position = input['gt_position']
        if "ec_position" in input:
            self.ec_position = input['ec_position'] 
        if "gaze" in input:
            self.output = input["gaze"]
        if "gaze_pt" in input:
            self.output = input["gaze_pt"]
        return self.net_input, self.output

    def forward(self, x):
        output = self.netGazeNetwork(x)
        return output

    def training_step(self, batch, batch_idx = 0):
        input, output = self.set_input(batch)
        output_pred = self(input)
        loss_train = self.criterionLoss(output, output_pred)
        # print(output, output_pred)
        batch_dictionary = {
            "loss": loss_train,
            'preds': output_pred.detach(), 'target': output.detach()
        }
        # if torch.isnan(input["face"].any()):
        #     print("?????????")
        #     print("ecs:",batch["ec"],"\n","idx:", batch["index"])
        #     print("?????????")
            
        if torch.any(loss_train > 1000):
            print(torch.where(abs(output[:,0]) > 1000))
            i = int(torch.where(abs(output[:,0]) > 1000)[0])
            print("idx:", batch["index"][0][i],batch["index"][2][i],"\n",input["ec"][i], output[i],output_pred[i])
            torchvision.utils.save_image(input["face"][i],"error_input.png", normalize=True)

        if batch_idx % self.opt.visual_freq == 0 or (loss_train > 1000):
            self.visual_names = ["face"]
            if self.opt.debug:
                self.visual_names += ["gt_position", "ec_position"]
            if self.opt.use_eyes:
                self.visual_names += ["leye", "reye"]

            self.get_current_visuals("train", batch_idx)

        return batch_dictionary

    def training_step_end(self,outputs):
        metric = self.train_metric(outputs['preds'], outputs['target'])
        # if torch.any(abs(metric > 100)):
        #     print(outputs['preds'], outputs['target'])
        # print(outputs['preds'].device)
        train_loss = torch.mean(outputs['loss'])
        self.log("train_loss",train_loss, on_step=True, on_epoch=True)

        self.log("train_error", self.train_metric, on_step=True, prog_bar=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        input, output = self.set_input(batch)
        output_pred = self(input)
        batch_dictionary = {
            'preds': output_pred, 'target': output
        }
        output_pred_np = output_pred.detach().cpu().numpy()
        self.pred_position = torch.zeros_like(self.gt_position)
        for i, pred in enumerate(output_pred_np):
            # print(pred, output[i])
            pred = [pred[0] / 53.15, pred[1] / 29.9]
            self.pred_position[i] = torch.tensor(draw_point(pred))
        if batch_idx % self.opt.visual_freq == 0:
            self.visual_names = ["face"]
            if self.opt.debug:
                self.visual_names += ["gt_position", "ec_position", "pred_position"]
            self.get_current_visuals("valid", batch_idx)

        return batch_dictionary

    def validation_step_end(self,outputs):
        self.valid_error = self.valid_metric(outputs['preds'], outputs['target'])
        self.log("val_error", self.valid_metric, on_step=True, on_epoch=True)
        self.log("hp_metric", self.valid_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.netGazeNetwork.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))

        return [optimizer], [scheduler]

