#![allow(unused_imports)]
use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
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

    let play: Play = support::get_ai_play("d1-e1");
    println!("{play:?}");

    println!("{} to {}", 2252, support::action_to_str(&2252));

    // let actions = (0..50).collect::<Vec<_>>();
    // for action in actions {
    //     println!("{} to {}",action, support::action_to_str(&action));
    // }

    // let actions = (2000..2100).collect::<Vec<_>>();
    // for action in actions {
    //     println!("{} to {}",action, support::action_to_str(&action));
    // }

    let mut nnmodel = CModule::load("./models/example.pt").unwrap();
    nnmodel.set_eval();

    let _result = mcts(&nnmodel, &game, 100);

}

