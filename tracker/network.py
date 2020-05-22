import os.path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from re3_utils.pytorch_util import pytorch_util_functions as pt_util
from re3_utils.pytorch_util.CaffeLSTMCell import CaffeLSTMCell


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> Norm -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.GroupNorm(32, out_channels)
        self.nonlinearity = nn.ELU(inplace=True)
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.nonlinearity(x)
        return x


class Re3NetBase(nn.Module):
    def __init__(self, device, args=None):
        super(Re3NetBase, self).__init__()
        self.device = device
        self.args = args
        self.learning_rate = None
        self.optimizer = None
        self.outputs = None

    def loss(self, outputs, labels):
        l1_loss = F.l1_loss(outputs, labels)
        return l1_loss

    def setup_optimizer(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0005)

    def update_learning_rate(self, lr_new):
        if self.learning_rate != lr_new:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_new
            self.learning_rate = lr_new

    def step(self, inputs, labels):
        self.optimizer.zero_grad()
        self.outputs = self(inputs)
        loss = self.loss(self.outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.data.cpu().numpy()[0]


class Re3Net(Re3NetBase):
    def __init__(self, device, lstm_size=1024, args=None):
        super(Re3Net, self).__init__(device, args)
        self.device = device
        self.lstm_size = lstm_size
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(3, 96, 11, stride=4, padding=0),
                nn.Conv2d(96, 256, 5, padding=2, groups=2),
                nn.Conv2d(256, 384, 3, padding=1),
                nn.Conv2d(384, 384, 3, padding=1, groups=2),
                nn.Conv2d(384, 256, 3, padding=1, groups=2),
            ]
        )
        self.lrn = nn.ModuleList(
            [
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            ]
        )

        self.conv_skip = nn.ModuleList([nn.Conv2d(96, 16, 1), nn.Conv2d(256, 32, 1), nn.Conv2d(256, 64, 1), ])
        self.prelu_skip = nn.ModuleList([torch.nn.PReLU(16), torch.nn.PReLU(32), torch.nn.PReLU(64)])

        self.fc6 = nn.Linear(74208, 2048)

        self.lstm1 = CaffeLSTMCell(2048, self.lstm_size)
        self.lstm2 = CaffeLSTMCell(2048 + self.lstm_size, self.lstm_size)

        self.lstm_state = None

        self.fc_output_out = nn.Linear(self.lstm_size, 4)

        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x if len(x.shape) == 4 else pt_util.remove_dim(x, 1)),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(
                    lambda x: pt_util.normalize(
                        x,
                        mean=np.array([123.151630838, 115.902882574, 103.062623801], dtype=np.float32)[
                             np.newaxis, np.newaxis, np.newaxis, :
                             ],
                    )
                ),
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            ]
        )

    def forward(self, input, lstm_state=None):
        batch_size = input.shape[0]
        input = self.transform(input).to(device=self.device)
        conv1 = self.conv[0](input)
        pool1 = F.relu(F.max_pool2d(conv1, (3, 3), stride=2))
        lrn1 = self.lrn[0](pool1)

        conv1_skip = self.prelu_skip[0](self.conv_skip[0](lrn1))
        conv1_skip_flat = pt_util.remove_dim(conv1_skip, [2, 3])

        conv2 = self.conv[1](lrn1)
        pool2 = F.relu(F.max_pool2d(conv2, (3, 3), stride=2))
        lrn2 = self.lrn[1](pool2)

        conv2_skip = self.prelu_skip[1](self.conv_skip[1](lrn2))
        conv2_skip_flat = pt_util.remove_dim(conv2_skip, [2, 3])

        conv3 = F.relu(self.conv[2](lrn2))
        conv4 = F.relu(self.conv[3](conv3))
        conv5 = F.relu(self.conv[4](conv4))
        pool5 = F.relu(F.max_pool2d(conv5, (3, 3), stride=2))
        pool5_flat = pt_util.remove_dim(pool5, [2, 3])

        conv5_skip = self.prelu_skip[2](self.conv_skip[2](conv5))
        conv5_skip_flat = pt_util.remove_dim(conv5_skip, [2, 3])

        skip_concat = torch.cat([conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat], 1)
        skip_concat = pt_util.split_axis(skip_concat, 0, -1, 2)
        reshaped = pt_util.remove_dim(skip_concat, 2)

        fc6 = F.relu(self.fc6(reshaped))

        if lstm_state is None:
            outputs1, state1 = self.lstm1(fc6)
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1))
        else:
            outputs1, state1, outputs2, state2 = lstm_state
            outputs1, state1 = self.lstm1(fc6, (outputs1, state1))
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1), (outputs2, state2))

        self.lstm_state = (outputs1, state1, outputs2, state2)

        fc_output_out = self.fc_output_out(outputs2)
        return fc_output_out


class Re3SmallNet(Re3NetBase):
    def __init__(self, device, lstm_size=512, args=None):
        super(Re3SmallNet, self).__init__(device, args)
        self.lstm_size = lstm_size

        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, padding=3, kernel_size=7, stride=4),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        )

        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x if len(x.shape) == 4 else pt_util.remove_dim(x, 1)),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(
                    lambda x: pt_util.normalize(
                        x,
                        mean=np.array([123.675, 116.28, 103.53])[np.newaxis, np.newaxis, np.newaxis, :],
                        std=np.array([58.395, 57.12, 57.375])[np.newaxis, np.newaxis, np.newaxis, :],
                    )
                ),
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            ]
        )

        self.fc6 = nn.Linear(50176, 2048)
        self.lstm1 = nn.LSTMCell(2048, self.lstm_size)
        self.lstm2 = nn.LSTMCell(2048 + self.lstm_size, self.lstm_size)
        self.fc_output = nn.Sequential(
            nn.Linear(self.lstm_size, self.lstm_size), nn.ELU(inplace=True), nn.Linear(self.lstm_size, 4)
        )
        self.learning_rate = None
        self.optimizer = None
        self.outputs = None
        self.lstm_state = None

    def forward(self, input, lstm_state=None):
        x = input.to(self.device, dtype=torch.float32)
        x = self.transform(x)
        x = self.feature_extractor(x)
        x = pt_util.split_axis(x, 0, -1, 2)
        x = pt_util.remove_dim(x, (2, 3, 4))

        fc6 = F.elu(self.fc6(x))

        if lstm_state is None:
            outputs1, state1 = self.lstm1(fc6)
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1))
        else:
            outputs1, state1, outputs2, state2 = lstm_state
            outputs1, state1 = self.lstm1(fc6, (outputs1, state1))
            outputs2, state2 = self.lstm2(torch.cat((fc6, outputs1), 1), (outputs2, state2))

        self.lstm_state = (outputs1, state1, outputs2, state2)

        output = self.fc_output(outputs2)
        return output
