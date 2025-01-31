#![allow(unused_imports)]
use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use tch::CModule;
mod hnefgame;
mod mcts;
mod support;
use crate::mcts::mcts;


fn main() {

    let game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");

    let mut nnmodel = CModule::load("./models/example.pt").unwrap();
    nnmodel.set_eval();

    let _result = mcts(&nnmodel, &game, 10);

}

