import torch
import torch.nn as nn
import numpy as np


# class Res_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
#         super(Res_Block, self).__init__()
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 1),
#                         nn.BatchNorm2d(out_channels),
#                         nn.ReLU())
#         self.conv2 = nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 1),
#                         nn.BatchNorm2d(out_channels))
#         self.downsample = downsample
#         self.tanh = nn.Tanh()
#         self.out_channels = out_channels
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


class Othello_QNet(nn.Module):
    def __init__(self, board_size, in_channels=4, hidden_channels=8):
        super(Othello_QNet, self).__init__()
        self.board_size = board_size
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels

        # Input shape: (3, board_size, board_size)
        self.f = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.Tanh(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.Tanh(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.Tanh(),
        )

        self.Conv_Policy = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        self.Softmax = nn.Softmax(dim=0)

        self.FC_Value = nn.Linear(self.hidden_channels * self.board_size**2, 1, bias=False)
    
    def forward(self, x):
        x = self.f(x)
        ac_probs = self.Conv_Policy(x)
        ac_probs = self.Softmax(torch.flatten(ac_probs)).reshape(-1, 1, self.board_size, self.board_size)
        value = self.FC_Value(torch.flatten(x))
        return value, ac_probs