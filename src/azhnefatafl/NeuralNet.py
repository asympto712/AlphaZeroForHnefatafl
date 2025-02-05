import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import *
from .taflNNet import TaflNNet as onnet

# This class is a direct import from Alpha-Zero-General
class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self):
        pass

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        pass

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
}

# game = {
#     'boardsize' : 7,
#     'actionsize' : actionsize,
#     }


class NNetWrapper(NeuralNet):
    def __init__(self, jit_model):
        self.nnet = jit_model
        # self.board_x, self.board_y = game['boardsize']
        # self.action_size = game['actionsize']

        if args["cuda"]:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, player, v)
        board should already be a matrix representation
        v should always be from the perspective of the attacker 
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args["epochs"]):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()

            batch_count = int(len(examples) / args["batch_size"])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args["batch_size"])
                boards, pis, players, vs = list(zip(*[examples[i] for i in sample_ids]))
                # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis))
                players = torch.BoolTensor([True if player == 1 else False for player in players])
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args["cuda"]:
                    boards, target_pis, players, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), players.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                pis, vs = self.nnet(boards, players)
                
                l_pi = self.loss_pi(target_pis, pis)
                l_v = self.loss_v(target_vs, vs)
                total_loss = l_pi + l_v

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args["cuda"]: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    # output is log_softmax
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
