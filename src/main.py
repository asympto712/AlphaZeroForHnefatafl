import azhnefatafl as azh
"""
See README.md for instruction. Your code interpreter might give you a warning. Ignore it.
"""
# from  _azhnefatafl import self_play_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
}


model = azh.TaflNNet(game, args)
scripted_model = torch.jit.script(model)
scripted_model.save("models/example.pt")
result = azh.self_play_function("models/example.pt", 1)

jit_model = torch.jit.load("models/example.pt")
model = azh.NNetWrapper(jit_model)
model.train(result)
