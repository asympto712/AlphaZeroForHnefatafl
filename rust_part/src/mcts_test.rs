#![allow(unused_imports)]

mod hnefgame;
mod mcts;
mod support;
mod mcts_par;

use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
use crate::hnefgame::board::state::BoardState;
use crate::hnefgame::board::state::BitfieldBoardState;
use crate::mcts::mcts;
use crate::mcts_par::{Tree, Node};
use crate::hnefgame::tiles::Tile;
use tch::CModule;
use std::str::FromStr;

fn main() {
    let mut game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");
    let board: BitfieldBoardState<u64> = BitfieldBoardState::from_display_str(
        ".K.....\n.......\n.......\n.......\n.......\n.......\n.......\n"
    ).unwrap();
    game.state.board = board;
    println!("Player: {:?}", game.state.side_to_play);
    println!("board: ");
    println!("{}", game.state.board);
}