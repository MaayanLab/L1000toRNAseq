import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim, output_dim_decoder, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        input_dim = input_shape[1]

        # Encoder (Downsampling)
        model = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ]

        # Transformation (Residual blocks)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(output_dim)]

        # Decoder (Upsampling)
        model += [
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_decoder),
        ]

        # Output layer
        # model += [nn.ReflectionPad2d(channels),
        #          nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        input_dim = input_shape[1]
        self.output_shape = (output_dim)

        # Extract features from generated sample
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )

    def forward(self, img):
        return self.model(img)



########################################
#        Step2: Extrapolation
########################################


class Extrapolator(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_dim):
        super(Extrapolator, self).__init__()

        input_dim = input_shape[1]

        # Encoder (Downsampling)
        model = []
        for i in range(len(hidden_dims)):
            if i == 0:
                model.append(nn.Linear(input_dim, hidden_dims[0]))
                model.append(nn.ReLU())
            
            else:
                model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                model.append(nn.ReLU())
            
        model.append(nn.Linear(hidden_dims[i], output_dim))
        # model.append(nn.ReLU())
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
