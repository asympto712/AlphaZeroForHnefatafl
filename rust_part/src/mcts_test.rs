#![allow(unused_imports)]

mod hnefgame;
mod mcts;
mod support;
mod mcts_par;
mod mcts_cmp;

use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
use crate::hnefgame::board::state::BoardState;
use crate::hnefgame::board::state::BitfieldBoardState;
use crate::hnefgame::tiles::Tile;
use crate::mcts::mcts;
use crate::mcts_par::{Tree, Node};
use crate::support::{action_to_str, board_to_matrix, get_ai_play};

use tch::CModule;
use std::str::FromStr;
use std::io::{self, Write};
use std::sync::Arc;

fn main() {

    let num_iter: usize = 1600;
    let num_workers: usize = 2;
    let c_puct: f32 = 1.0;
    let alpha: f64 = 0.4;
    let eps: f32 = 0.25;

    let mut game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");
    let board: BitfieldBoardState<u64> = BitfieldBoardState::from_display_str(
        ".K.....\n.......\n.......\n.......\n.......\n.......\n.......\n"
    ).unwrap();
    game.state.board = board;
    game.state.side_to_play = hnefgame::pieces::Side::Defender;
    println!("Player: {:?}", game.state.side_to_play);
    println!("board: ");
    println!("{}", game.state.board);

    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    let mut nnmodel = CModule::load_on_device("./agents/300lim_4gen_plain_vanilla/models/gen4.pt", device).unwrap();
    nnmodel.set_eval();
    let nnmodel = Arc::new(nnmodel);

    for mcts_alg in mcts_cmp::MCTSAlg::iter() {
        println!("MCTS algorithm: {:?}", mcts_alg);
        let game = game.clone();
        let policy = mcts_cmp::mcts_do_alg(&mcts_alg, Arc::clone(&nnmodel), &game, num_iter, num_workers, c_puct, alpha, eps);
        let mut policy_indices: Vec<_> = policy.iter().enumerate().collect();
        policy_indices.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let top_5_indices: Vec<u32> = policy_indices.iter().take(5).map(|&(index, _)| index as u32).collect();
        for action in top_5_indices {
            let mut copied_game = game.clone();
            let str_action: &str = &action_to_str(&action);
            println!("action: {}, prob: {}", str_action, policy[action as usize]);
            let play = get_ai_play(str_action);
            let result = copied_game.do_play(play);
            match result {
                Ok(_) => {
                    println!("board after play: ");
                    println!("{}", copied_game.state.board);
                }
                Err(_) => ()
            }
        }
    }
}