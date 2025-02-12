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

use tch::CModule;

fn main() {
    
}