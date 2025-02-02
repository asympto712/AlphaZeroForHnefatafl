#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::hnefgame::game::state::GameState;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::game::{SmallBasicGame, Game};
use crate::support::{action_to_str, board_to_matrix, get_ai_play,write_to_file};
use crate::hnefgame::game::GameOutcome::{Win, Draw};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::board::state::BoardState;
use crate::hnefgame::play::Play;
use rand::prelude::*;
use rand::rng;
use crate::mcts::mcts;
use crate::hnefgame::preset::{boards, rules};
use rand::distr::weighted::WeightedIndex;
use tch::CModule;

use std::thread;
use termion::input::TermRead;
use std::io::{self,Write};
use std::sync::mpsc;


fn generate_training_example<T: BoardState>(
    game_state_history: &Vec<GameState<T>>,
    policy_history: &Vec<Vec<f32>>,
    final_outcome: i32,
) -> Vec<(Vec<Vec<u8>>, Vec<f32>, i32, f32)> {
    let mut training_examples = Vec::new();
    let reward = final_outcome as f32;


    for (state, policy) in game_state_history.iter().zip(policy_history.iter()) {

        let no_def = BoardState::count_pieces(&state.board, Side::Defender) as f32;
        let no_att = BoardState::count_pieces(&state.board, Side::Attacker) as f32;



        training_examples.push((

            board_to_matrix(state),
            policy.clone(),

            match state.side_to_play {
                Side::Attacker => 1,
                Side::Defender => -1,
            },
            // Reward is given as +1 if attacker wins and -1 if defender wins. Attacker is favored when reward is positive.
            // We take game outcome and add how many defenders were captured (benefits attacker) 
            // and subtract how many attackers were captured (benefits defender)

            reward + 0.2 * (5.0 - no_def) - 0.1 * (8.0 - no_att)
            // reward
            // switch lines above for pure win/loss reward
        ));
    }
    training_examples
}

// For testing
use std::time::Instant;


pub fn self_play(nnmodel: CModule, no_games: i32) -> Result<Vec<(Vec<Vec<u8>>, Vec<f32>, i32, f32)>, String> {

    let mut training_data = Vec::new();

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let stdin = io::stdin();
        for c in stdin.keys() {
            let c = c.unwrap();
            if c == termion::event::Key::Char('e') {
                tx.send("exit").unwrap();
                break;
            }
        }
    });

    // ctrlc::set_handler(move || tx.send(()).expect("Could not send signal on channel."))
    // .expect("Error setting Ctrl-C handler");

    for i in 0..no_games {
        println!("Game number: {}", i);
    
    // Create new game
        let mut game: SmallBasicGame = Game::new(
            rules::KOCH,
            boards::BRANDUBH,
        ).expect("Could not create game.");

        let mut policy_history = Vec::new();

        loop {
            let move_time = Instant::now();

            // check for exit string
            if let Ok(msg) = rx.try_recv() {
                if msg == "exit" {
                    println!("Exiting game...");
                    return Err("Exit command received".to_string());
                }
            }


            let player = game.state.side_to_play;
            println!("Player: {:?}", player);

            println!("Board:");
            println!("{}", game.state.board);

            let policy = mcts(&nnmodel, &game, 100);
            policy_history.push(policy.clone());

            let mut rng = rng();
            let dist = WeightedIndex::new(&policy).expect("Invalid distribution");
            let action = dist.sample(&mut rng) as u32;
            let str_action: &str = &action_to_str(&action);
            let play = get_ai_play(str_action);
            
            // For testing purpose. TEMP
            println!("{}", str_action);

            match game.do_play(play) {
                Ok(status) => {
                    println!("Move took: {:?}", move_time.elapsed());

                    if let Over(outcome) = status {
                        match outcome {
                            Draw(reason) => {
                                println!("Game over. Draw {reason:?}.");
                                let mut training_examples = generate_training_example(&game.state_history, &policy_history, 0);
                                training_data.append(&mut training_examples);
                                break;
                            }
                            Win(reason, side) => {
                                let outcome_value = match side {
                                    Side::Attacker => 1,
                                    Side::Defender => -1,
                                };
                                println!("Game over. Winner is {side:?} ({reason:?}).");
                                let mut training_examples = generate_training_example(&game.state_history, &policy_history, outcome_value);
                                training_data.append(&mut training_examples);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Invalid move ({e:?}). Try again.");
                    //let max_index = policy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(index, _)| index as u32).expect("Policy vector is empty");
                    //let play = get_ai_play(&action_to_str(&action));
                    continue
                }
            }
        }

    }

    
    // // Write serialized data to file
    // let file_path = "data/training_data.txt";
    // for (board, policy, side, outcome) in &training_data {
    //     write_to_file(file_path, &board, &policy, *side, *outcome);
    // }
    Ok(training_data)
}