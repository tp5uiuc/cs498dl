import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
from typing import Any

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MySimpleClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.ReLU(inplace=True),
            maxpool,
        )

        self.bridge = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 26 * 26, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x


# replace the 5x5 layer with 2 3x3 layers with maxpools taking inspiration from
# VGGnet
# increase the channel width from start to end rather than decreasing it
# once again inspired from VGGnet
class MyClassifierV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=True),
            # maxpool,
        )

        self.bridge = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 26 * 26, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x


# The previous one barely learns, maybe because of vanishing gradients in the
# deep layer. As a cure, add batchnorm
# Add batchnorm layers for the features, inspired by GoogleNet Inception v3
class MyClassifierV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            # maxpool, # don't divide by 2 here, too low res
        )

        self.bridge = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 26 * 26, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x


# After batchnorming, the performance improves a lot. But the gap between
# train and test seems to go up
# maybe its because the model cannot generalize well (doesn't have enough
# generalization capacity). In the context,
# 256 -> 16 in the last layer using 1x1 convolutions seem aggressive
# so we map it to 256 -> 128 in the last layer and make linear layers wider
# to accommodate these additional degrees of freedom
class MyClassifierV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # maxpool, # don't divide by 2 here, too low res
        )

        self.bridge = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 26 * 26, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=420),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=420, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x


# From the last attempt, while the mAP goes up, the generalization is very poor
# Also the degrees of freedom in the fully connected layers shoot up like crazy
# So maybe our hypothesis in 3->4 transition is wrong.
# We addressing both these issues by adding an average pooling layer
# to replace the last convolutional layers, Inspired by NiN.
# we retain the classifier dimensions from V2 however
class MyClassifierV4(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # 6 because 26/4.5 ~= 5, so that we maintain 3x the
            # complexity of the original model
            nn.AdaptiveAvgPool2d((5, 5))
            # maxpool, # don't divide by 2 here, too low res
        )

        self.bridge = nn.Flatten()

        # original model params in linear classifer 1309989
        # this model params in linear classifer 3717165
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 5 * 5, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=420),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=420, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x


# V4 seems to be on track
# but we note that no regularization is added for the linear layers
# to fix this we add dropout inspired by AlexNet
class MyClassifierV5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        dropout = nn.Dropout(0.5)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # 6 because 26/4.5 ~= 5, so that we maintain 3x the
            # complexity of the original model
            nn.AdaptiveAvgPool2d(output_size=(5, 5))
            # maxpool, # don't divide by 2 here, too low res
        )

        self.bridge = nn.Flatten()

        # original model params in linear classifer 1309989
        # this model params in linear classifer 3717165
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 5 * 5, out_features=1024),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(in_features=1024, out_features=420),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(in_features=420, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x


# V5 seems to be more on on track than V4
# But its still slightly below what the expected mAPs are in the piazza post
# to fix that, I give more latitude to the linear layer
# from 1024 -> 420 -> 105 -> 21
class MyClassifierV6(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        dropout = nn.Dropout(0.5)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            maxpool,
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # 6 because 26/4.5 ~= 5, so that we maintain 3x the
            # complexity of the original model
            nn.AdaptiveAvgPool2d(output_size=(5, 5))
            # maxpool, # don't divide by 2 here, too low res
        )

        self.bridge = nn.Flatten()

        # original model params in linear classifer 1309989
        # this model params in linear classifer 3717165
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 5 * 5, out_features=1024),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(in_features=1024, out_features=420),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(in_features=420, out_features=105),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(in_features=105, out_features=NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.bridge(x)
        x = self.classifier(x)
        return x
