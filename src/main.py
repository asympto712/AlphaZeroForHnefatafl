import azhnefatafl as azh
"""
See README.md for instruction. Your code interpreter might give you a warning. Ignore it.
"""
<<<<<<< HEAD
=======
# from  _azhnefatafl import self_play_function

>>>>>>> f65eafa0d13d4631a28958d417bfaa503e508930
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
<<<<<<< HEAD
from collections import deque
=======
>>>>>>> f65eafa0d13d4631a28958d417bfaa503e508930

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
<<<<<<< HEAD
    'maxlen': 10000,
    'numIter': 10,
    'numGamesPerIter': 100,
}


# model = azh.TaflNNet(game, args)
# scripted_model = torch.jit.script(model)
# scripted_model.save("models/example.pt")
# result = azh.self_play_function("models/example.pt", 1)

# print(type(result))

# jit_model = torch.jit.load("models/example.pt")
# model = azh.NNetWrapper(jit_model)
# model.train(result)

wrapper = azh.NNetWrapper(args, game)
wrapper.learn()

# self-play -> train loop
#TODO: Implement stop logic
# loop {
#     #self-play
#     result = azh.self_play_function(model_path, 10)
#     #train & save
#     jit_model = torch.jit.load(model_path)
#     model = azh.NNetWrapper(jit_model)
#     model.train(result)
#     new_model_path = model.save_checkpoint()
#     model_path = new_model_path
# }
=======
}


model = azh.TaflNNet(game, args)
scripted_model = torch.jit.script(model)
scripted_model.save("models/example.pt")
_ = azh.self_play_function("models/example.pt", 1)

result = azh.read_training_data("data/training_data.txt")

jit_model = torch.jit.load("models/example.pt")
model = azh.NNetWrapper(jit_model)
model.train(result)
>>>>>>> f65eafa0d13d4631a28958d417bfaa503e508930
