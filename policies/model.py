import torch
import torch.nn as nn
import numpy as np

class Othello_QNet(nn.Module):
    def __init__(self, board_size, in_channels=4, hidden_channels=8):
        super(Othello_QNet, self).__init__()
        self.board_size = board_size
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.actions = [i for i in range(self.board_size**2)]

        # Input shape: (4, board_size, board_size)
        self.f = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
        )

        self.Conv_Policy = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        self.Softmax = nn.Softmax(dim=1)

        self.FC_Value = nn.Linear(self.hidden_channels * self.board_size**2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.board_size, self.board_size)
        x = self.f(x)

        # Compute ac_probs
        ac_probs = self.Conv_Policy(x).reshape(-1, self.board_size**2)
        ac_probs = self.Softmax(ac_probs)
        
        # Compute Q value
        value = self.FC_Value(x.view(-1, self.hidden_channels * self.board_size**2))
        value = self.sigmoid(value)

        return value, ac_probs

    def get_action(self, obs):
        obs = obs.reshape(self.in_channels, self.board_size, self.board_size)
        _, ac_probs = self.forward(obs)
        legal_moves_mask = torch.flatten(obs[-1])
        ac_probs = (ac_probs * legal_moves_mask).reshape(-1)
        ac_probs = ac_probs.detach().numpy()
        ac_probs = ac_probs / np.sum(ac_probs)

        np.random.seed(np.random.randint(0, 1000))
        action = np.random.choice(self.actions, size=1, p=ac_probs)
        return action.item()

