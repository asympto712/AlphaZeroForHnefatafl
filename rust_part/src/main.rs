use hnefatafl::game::{Game, SmallBasicGame};
use hnefatafl::game::GameOutcome::{Draw, Win};
use hnefatafl::game::GameStatus::Over;
use std::any::type_name;

mod support;
use crate::support::get_all_possible_moves;
use crate::support::get_play;

use mcts;




fn main() {
    println!("hnefatafl-rs demo");
    let mut game: SmallBasicGame = Game::new(
        hnefatafl::preset::rules::KOCH,
        hnefatafl::preset::boards::BRANDUBH,
    ).expect("Could not create game.");


    loop {


        println!("Board:");
        println!("{}", game.state.board);

        println!("{:?} to play.", game.state.side_to_play);

        //println!("Possible moves: {:?}", get_all_possible_moves(&game));

        let play = match get_play() {
            Some(play) => play,
            None => {
                println!("Exiting the game.");
                return;
            }
        };

        

        match game.do_play(play) {
            Ok(status) => {
                if let Over(outcome) = status {
                    match outcome {
                        Draw(reason) => println!("Game over. Draw {reason:?}."),
                        Win(reason, side) => println!("Game over. Winner is {side:?} ({reason:?})."),
                    }
                    println!("Final board:");
                    println!("{}", game.state.board);
                    return
                }
            },
            Err(e) => println!("Invalid move ({e:?}). Try again.")
        }
    }
}