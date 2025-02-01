import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
from typing import Dict
from .utils import *

# game = {
#     'boardsize' : 7,
#     'actionsize' : actionsize,
#     }

# args = {
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': torch.cuda.is_available(),
#     'num_channels': 512,
# }

class TaflNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game['boardsize']
        self.action_size = game['actionsize']
        self.lr: float = args['lr']
        self.dropout: float = args['dropout']
        self.epochs: int = args['epochs']
        self.batch_size: int = args['batch_size']
        self.cuda: bool = args['cuda']
        self.num_channels: int = args['num_channels']

        super(TaflNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, self.action_size)

        self.fc5 = nn.Linear(512, 1)

    def forward(self, s: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        # pre_pi = self.fc3(s) if cond else self.fc4(s) 
        pre_pi = torch.where(cond.view(-1,1), self.fc3(s), self.fc4(s))                                             # batch_size x action_size
        pre_v = self.fc5(s)  
        pi = torch.log_softmax(pre_pi, 1)
        v = torch.tanh(pre_v)                                                                         # batch_size x 1

        return (pi, v)
    
# example:
# model = TaflNNet(game, args)
# scripted_model = torch.jit.script(model)
# scripted_model.save("scripted_model.pt")
# loaded_model = torch.jit.load("scripted_model.pt")
    

