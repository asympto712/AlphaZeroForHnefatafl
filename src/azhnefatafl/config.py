import torch

Game = {
    'boardsize' : (7, 7),
    'actionsize' : 49 * 49,
}

Args = {
    'lr': 0.2,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'maxlen': 50000,
    'numGamesPerGen': 100,
    'mcts': 400,
    'mcts_alg': "mcts_par_mcts_root_par",
    'num_workers': 8,
    'c_puct': 0.10,
    'alpha': 0.3,
    'eps': 0.25,
}

"""
mcts_alg can be chosen from: mcts_mcts, mcts_par_mcts_notpar, mcts_par_mcts_par, mcts_par_mcts_root_par
For details, see rust_part/src/duel.rs
"""