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
    'mcts_alg': "mcts_par_mcts_par",
    'num_workers': 4,
    'c_puct': 0.3,
    'alpha': 0.3,
    'eps': 0.25,
}

"""
mcts_alg can be chosen from: mcts_mcts, mcts_par_mcts_notpar, mcts_par_mcts_par, mcts_par_mcts_root_par
For details, see rust_part/src/duel.rs
"""