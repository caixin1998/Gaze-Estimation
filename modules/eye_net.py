"""Copyright 2020 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging

import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet
half_pi = 0.5 * np.pi

class eyenet_single(nn.Module):
    def __init__(self, opt):
        super(eyenet_single, self).__init__()

        # CNN backbone (ResNet-18 with instance normalization)
        self.cnn_layers = ResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                                 num_classes=opt.ngf,
                                 norm_layer=nn.InstanceNorm2d)
        
        self.opt = opt
        num_features = opt.ngf
        self.fc_common = nn.Sequential(
            nn.Linear(opt.ngf + (2 if opt.eye_net_use_head_pose_input else 0),
                      opt.ngf),
            nn.SELU(inplace=True),
            nn.Linear(opt.ngf, opt.ngf),
        )

        # FC layers
        self.fc_to_gaze = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, 2, bias=False),
            nn.Tanh(),
        )
        # self.fc_to_pupil = nn.Sequential(
        #     nn.Linear(num_features, num_features),
        #     nn.SELU(inplace=True),
        #     nn.Linear(num_features, 1),
        #     nn.ReLU(inplace=True),
        # )

        # Set gaze layer weights to zero as otherwise this can
        # explode early in training
        nn.init.zeros_(self.fc_to_gaze[-2].weight)

    def forward(self, input_dict, side):
        # Pick input image
        input_image = input_dict[side[0] + "eye"]
        # Compute CNN features
        initial_features = self.cnn_layers(input_image)

        # Process head pose input if asked for
        if self.opt.eye_net_use_head_pose_input:
            initial_features = torch.cat([initial_features, input_dict[side + '_h']], axis=1)
        initial_features = self.fc_common(initial_features)
        # Final prediction
        gaze_prediction = half_pi * self.fc_to_gaze(initial_features)
        # For gaze, the range of output values are limited by a tanh and scaling
        return gaze_prediction
        # Estimate of pupil size
        # output_dict[side + '_pupil_size'] = pupil_size.reshape(-1)

        # If network frozen, we're gonna detach gradients here
        # if config.eye_net_frozen:
        #     output_dict[side + '_g_initial'] = output_dict[side + '_g_initial'].detach()

class EyeNet(nn.Module):
    def __init__(self, opt):
        super(EyeNet, self).__init__()
        self.left_net = eyenet_single(opt)
        self.right_net = eyenet_single(opt)
    def forward(self, input_dict):
        output_dict = {}
        output_dict["left_g"] = self.left_net(input_dict, "left")
        output_dict["right_g"] = self.left_net(input_dict, "right")
        return output_dict
