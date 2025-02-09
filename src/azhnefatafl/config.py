import torch

Game = {
    'boardsize' : (7, 7),
    'actionsize' : 49 * 49,
}

Args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'maxlen': 20000,
    'numGamesPerGen': 100,
    'mcts': 100,
}