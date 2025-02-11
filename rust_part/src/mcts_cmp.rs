
use crate::hnefgame::game::Game;
use crate::hnefgame::board::state::BoardState;
use crate::mcts;
use crate::mcts_par;
use tch::CModule;
use std::sync::Arc;

#[derive(Debug)]
pub enum MCTSAlg {
    MctsMcts,
    MctsParMctsNotpar,
    MctsParMctsPar,
    MctsParMctsRootPar,
}

impl std::str::FromStr for MCTSAlg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mcts_mcts" => Ok(MCTSAlg::MctsMcts),
            "mcts_par_mcts_notpar" => Ok(MCTSAlg::MctsParMctsNotpar),
            "mcts_par_mcts_par" => Ok(MCTSAlg::MctsParMctsPar),
            "mcts_par_mcts_root_par" => Ok(MCTSAlg::MctsParMctsRootPar),
            _ => Err("Invalid MCTS algorithm".to_string()),
        }
    }
}

pub fn mcts_do_alg<T: BoardState + Send + 'static>(
    mcts_alg: &MCTSAlg,
    nnmodel: Arc<CModule>,
    game: &Game<T>,
    num_iter:usize,
    num_workers: usize,
    c_puct: f32,
    alpha: f64,
    eps: f32) 
    -> Vec<f32>{

    match mcts_alg {
        MCTSAlg::MctsMcts => {
            let policy = mcts::mcts(&nnmodel, game, num_iter);
            policy
        },
        MCTSAlg::MctsParMctsNotpar => {
            let policy = mcts_par::mcts_notpar(&nnmodel, game, num_iter, c_puct, alpha, eps);
            policy
        },
        MCTSAlg::MctsParMctsPar => {
            let policy = mcts_par::mcts_par(nnmodel, game, num_iter, num_workers, c_puct, alpha, eps);
            policy
        },
        MCTSAlg::MctsParMctsRootPar => {
            let policy = mcts_par::mcts_root_par(nnmodel, game, num_iter, num_workers, c_puct, alpha, eps);
            policy
        },
    }
}