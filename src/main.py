import azhnefatafl as azh
"""
See README.md for instruction. Your code interpreter might give you a warning. Ignore it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

game = azh.Game
args = azh.Args

"""
To create a new wrapper instance and train it from scratch..
wrapper = azh.NNetWrapper(args, game)
wrapper.learn(verbose=False, maxgen=None) <- your choice. True will display the moves, boards,.. False will not

"""

"""
To load a wrapper and start training where you left it..
wrapper = azh.load_wrapper(WrapperName)
wrapper.learn(verbose=False, maxgen=None) <- your choice.
"""

"""
To change one of the args..
wrapper.change_arg(ArgName, newvalue)
This will log the change.
"""

"""
To visualize the training process via tensorboard..
In your separate terminal, open the virtual env and
tensorboard --logdir=agents/(name of the wrapper that you want to see)/vcycle
Then go to whatever localhost they say..
If you get the error: No module named 'imghdr'
Try python -m pip install standard-imghdr
"""

# wrapper = azh.NNetWrapper(args, game)
# wrapper.learn(verbose=False) #your choice

wrapper = azh.NNetWrapper(args, game)
wrapper.learn("agents/Musk/models/gen1.pt", "agents/Musk/train_examples/gen0.npz", verbose = True, maxgen = 3)
wrapper.save_itself()

