import azhnefatafl as azh
"""
See README.md for instruction. Your code interpreter might give you a warning. Ignore it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

game = {
    'boardsize' : (7, 7),
    'actionsize' : 49 * 49,
}

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'maxlen': 20000,
    'numGamesPerGen': 1,
    'mcts': 100,
}

# wrapper = azh.NNetWrapper(args, game)
# wrapper.learn(verbose=False) #your choice
# wrapper.save_itself()

wrapper = azh.load_wrapper("test6")
wrapper.change_arg("batch_size", 20)
wrapper.change_arg("numGamesPerGen", 2)
wrapper.learn(verbose = True)


"""
If you want to pick up where you left,
wrapper = azh.load_wrapper(wrapper_name)
wrapper.learn()
It automatically loads the latest model and train examples
"""
