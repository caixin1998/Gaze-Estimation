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
import time
from .base_model import BaseModel
from captum.attr import *
from . import networks
from pytorch_lightning import LightningModule
import torchvision
# from pytorch_grad_cam import * 
from metrics import MeanDistanceError,MeanAngularError
from util.data_util import draw_point, draw_gaze, tensor2im, grayarray2heatmaptensor
from util.eve_util import to_screen_coordinates, calculate_combined_gaze_direction
import cv2 as cv
import numpy as np
class EVEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--use_eyes', action='store_true', default=False, help='use two eyes for our model.')
        parser.add_argument('--write_features', action='store_true', default=False, help='save features to npy file.')

        parser.add_argument('--eye_net_use_head_pose_input', action='store_true', default=False, help='use head_pose for eye model.')




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
        
        self.data_time = time.time()
        self.train_time = time.time()
        
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
       
        self.netGazeNetwork = networks.define_EyeNetwork(opt)


        if opt.criterion == "mse":
            self.criterionLoss = torch.nn.MSELoss()
        elif opt.criterion == "l1":
            self.criterionLoss = torch.nn.L1Loss()
        elif opt.criterion == "smoothl1":
            self.criterionLoss = torch.nn.SmoothL1Loss()

        self.train_distance_metric = MeanDistanceError(dist_sync_on_step=True)
        self.valid_distance_metric = MeanDistanceError(dist_sync_on_step=True)

        self.train_angular_metric = MeanAngularError(dist_sync_on_step=True)
        self.valid_angular_metric = MeanAngularError(dist_sync_on_step=True)

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
        if "leye" in input and "reye" in input:
            self.eyes = torch.cat((input['reye'],input['leye']), axis = -1)
        
        if "left_h" in input:
            self.net_input["left_h"] = input["left_h"]
        if "right_h" in input:
            self.net_input["left_h"] = input["left_h"]

        if "gt_position" in input:
            self.gt_position = input['gt_position']
        
        self.output = {}
        
        if "left_g_tobii" in input:
            self.output["left_g"] = input["left_g_tobii"]
        if "right_g_tobii" in input:
            self.output["right_g"] = input["right_g_tobii"]
   
        return self.net_input, self.output


    def forward(self, x):
        output = self.netGazeNetwork(x)
        return output


    def training_step(self, batch, batch_idx = 0):
        input, output = self.set_input(batch)
        output_pred = self(input)
        self.calculate_additional_labels(batch)
        self.calculate_g_with_two_eyes(batch,output)

        if self.opt.camera_frame_type == "eyes":
            loss_train = self.criterionLoss(output["left_g"] * batch["left_g_tobii_validity"].repeat(2,1).T , output_pred["left_g"] * batch["left_g_tobii_validity"].repeat(2,1).T) + self.criterionLoss(output["right_g"] * batch["right_g_tobii_validity"].repeat(2,1).T, output_pred["right_g"] * batch["right_g_tobii_validity"].repeat(2,1).T)
        
            self.calculate_g_with_two_eyes(batch,output_pred)
        else:
            loss_train = self.criterionLoss(output["g"] * batch["g_validity"].repeat(2,1).T, output_pred["g"] * batch["g_validity"].repeat(2,1).T)

        validity = batch["left_g_tobii_validity"] * batch["right_g_tobii_validity"]

        batch_dictionary = {
            "loss": loss_train,
            'preds': output_pred, 'target': output, 'validity' : validity,
        }
        return batch_dictionary

    def training_step_end(self,outputs):
        metric = self.train_angular_metric(outputs['preds']["g"].detach() * outputs["validity"].repeat(2,1).T, outputs['target']["g"] * outputs["validity"].repeat(2,1).T)

        train_loss = torch.mean(outputs['loss'])
        self.log("train_loss",train_loss, on_step=True, on_epoch=True, sync_dist=True)

        self.log("train_error", self.train_angular_metric, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        return train_loss
    
    def on_validation_start(self):
        
        if self.opt.camera_frame_type == "eyes":
            self.visual_names = ["eyes", "eyes_pred_gt"]
        else:
            self.visual_names = ["face"]
            self.visual_names += ["face_pred_gt"]


        if self.opt.write_features:
            self.features = np.zeros((0,2048))

    def validation_step(self, batch, batch_idx):
        input, output = self.set_input(batch)

        output_pred = self(input)
        self.calculate_additional_labels(batch)
        self.calculate_g_with_two_eyes(batch,output)

        validity = batch["g_validity"]

        batch_dictionary = {
            'preds': output_pred, 'target': output, 'validity': validity
        }
        # print(batch.keys())
        if self.opt.camera_frame_type == "eyes":
            self.calculate_g_with_two_eyes(batch, output_pred)
        else:
            self.calculate_pog_with_g(batch,output_pred)
        # print(output_pred['PoG_cm'][0], output['PoG_cm'][0])
        if batch_idx % self.opt.visual_freq == 0:
            
            output_pred_np = output_pred['g'].detach().cpu().numpy()
            gts = output['g'].cpu().numpy()

            if self.opt.camera_frame_type == "eyes":
                self.eyes_pred_gt = torch.zeros_like(self.eyes[:64])
                
                for i, pred in enumerate(output_pred_np[:64]):
                    self.eyes_pred_gt[i] = torch.tensor(draw_gaze(gts[i], pred, image_in = tensor2im(self.eyes[i])))
            else:
                self.face_pred_gt = torch.zeros_like(self.face[:64])
                for i, pred in enumerate(output_pred_np[:64]):
                    self.face_pred_gt[i] = torch.tensor(draw_gaze(gts[i], pred, image_in = tensor2im(self.face[i])))

            self.get_current_visuals("valid", batch_idx)  
        return batch_dictionary

    def validation_step_end(self,outputs):
        if not self.opt.write_features:
            self.valid_error = self.valid_angular_metric(outputs['preds']['g'] * outputs["validity"].repeat(2,1).T, outputs['target']['g'] * outputs
            ["validity"].repeat(2,1).T)
            self.valid_distance_error = self.valid_distance_metric(outputs['preds']['PoG_cm'] * outputs["validity"].repeat(2,1).T, outputs['target']['PoG_cm'] * outputs
            ["validity"].repeat(2,1).T)
    
            self.log("val_error", self.valid_angular_metric, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val_distance_error", self.valid_distance_metric, on_step=True, on_epoch=True, sync_dist=True)
            self.log("hp_metric", self.valid_angular_metric, sync_dist=True)
        else:
            # print(outputs['preds'].detach().cpu().numpy().shape)
            self.features = np.concatenate((self.features,outputs['preds'].detach().cpu().numpy()))  
         
    def on_validation_epoch_end(self):
        if self.opt.write_features:
            print(self.features.shape)
            np.save(self.opt.dataset + "_features.npy", self.features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.netGazeNetwork.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))

        return [optimizer], [scheduler]


    def calculate_g_with_two_eyes(self, input, output, input_suffix="_tobii", output_suffix = ""):
        # Step 1) Calculate PoG from given gaze
        for side in ('left', 'right'):
            origin = input[side + '_o']
               
            direction = output[side + '_g' + output_suffix]
            rotation = input[side + '_R']
                        
            PoG_mm, PoG_px = to_screen_coordinates(origin, direction, rotation, input)
            output[side + '_PoG_cm' + output_suffix] = 0.1 * PoG_mm
            output[side + '_PoG_px' + output_suffix] = PoG_px

        # Step 1b) Calculate average PoG
        output['PoG_px' + output_suffix] = torch.mean(torch.stack([
            output['left_PoG_px' + output_suffix],
            output['right_PoG_px' + output_suffix],
        ], axis=-1), axis=-1)
        output['PoG_cm' + output_suffix] = torch.mean(torch.stack([
            output['left_PoG_cm' + output_suffix],
            output['right_PoG_cm' + output_suffix],
        ], axis=-1), axis=-1)
        output['PoG_mm' + output_suffix] = \
            10.0 * output['PoG_cm' + output_suffix]

        # Step 1b) Calculate the combined gaze (L/R)

        output['g' + output_suffix] = \
            calculate_combined_gaze_direction(
                input['o'],
                output['PoG_mm' + output_suffix],
                input['left_R'],  # by definition, 'left_R' == 'right_R'
                input['camera_transformation'],
            )
    
    def calculate_pog_with_g(self, input, output, output_suffix = ""):
        # Step 1) Calculate PoG from given gaze
        
        origin = input['o']
               
        direction = output['g' + output_suffix]
        rotation = input['left_R']
                        
        PoG_mm, PoG_px = to_screen_coordinates(origin, direction, rotation, input)
        output['PoG_cm' + output_suffix] = 0.1 * PoG_mm
        output['PoG_px' + output_suffix] = PoG_px

        output['PoG_mm' + output_suffix] = \
            10.0 * output['PoG_cm' + output_suffix]





    def calculate_additional_labels(self, full_input_dict, current_epoch=None):
        # sample_entry = next(iter(full_input_dict.values()))
        # batch_size = sample_entry.shape[0]
        

        # PoG in mm
        for side in ('left', 'right'):
            if (side + '_PoG_tobii') in full_input_dict:
                full_input_dict[side + '_PoG_cm_tobii'] = torch.mul(
                    full_input_dict[side + '_PoG_tobii'],
                    0.1 * full_input_dict['millimeters_per_pixel'],
                ).detach()
                full_input_dict[side + '_PoG_cm_tobii_validity'] = \
                    full_input_dict[side + '_PoG_tobii_validity']

        # Fake kappa to be used during training
        # Mirror the yaw angle to handle different eyes
        if 'left_o' in full_input_dict:
            full_input_dict['o'] = torch.mean(torch.stack([
                full_input_dict['left_o'], full_input_dict['right_o'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['o_validity'] = full_input_dict['left_o_validity']

        if 'left_PoG_tobii' in full_input_dict:
            # Average of left/right PoG values
            full_input_dict['PoG_px_tobii'] = torch.mean(torch.stack([
                full_input_dict['left_PoG_tobii'],
                full_input_dict['right_PoG_tobii'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['PoG_cm_tobii'] = torch.mean(torch.stack([
                full_input_dict['left_PoG_cm_tobii'],
                full_input_dict['right_PoG_cm_tobii'],
            ], axis=-1), axis=-1).detach()
            full_input_dict['PoG_px_tobii_validity'] = (
                full_input_dict['left_PoG_tobii_validity'].bool() &
                full_input_dict['right_PoG_tobii_validity'].bool()
            ).detach()
            full_input_dict['PoG_cm_tobii_validity'] = full_input_dict['PoG_px_tobii_validity']

        # 3D gaze direction for L/R combined gaze
        # print("input['camera_transformation'].shape:",full_input_dict['camera_transformation'].shape)
        if 'PoG_cm_tobii' in full_input_dict:
            full_input_dict['g'] = calculate_combined_gaze_direction(
                    full_input_dict['o'],
                    10.0 * full_input_dict['PoG_cm_tobii'],
                    full_input_dict['left_R'],
                    full_input_dict['camera_transformation'])
       
            full_input_dict['g_validity'] = full_input_dict['PoG_cm_tobii_validity']