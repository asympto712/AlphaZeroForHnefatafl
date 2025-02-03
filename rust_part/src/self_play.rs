#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
use crate::hnefgame::game::state::GameState;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::game::{SmallBasicGame, Game};
use crate::support::{action_to_str, board_to_matrix, get_ai_play,write_to_file};
use crate::hnefgame::game::GameOutcome::{Win, Draw};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
use rand::prelude::*;
use rand::rng;
use crate::mcts::mcts;
use crate::hnefgame::preset::{boards, rules};
use crate::hnefgame::board::state::BoardState;
use rand::distr::weighted::WeightedIndex;
use tch::CModule;
use std::time::Instant;


fn generate_training_example<T: BoardState>(
    game_state_history: &Vec<GameState<T>>,
    policy_history: &Vec<Vec<f32>>,
    final_outcome: i32,
) -> Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)> {
    let mut training_examples = Vec::new();

    for (state, policy) in game_state_history.iter().zip(policy_history.iter()) {
        training_examples.push((
            board_to_matrix(state),
            policy.clone(),
            match state.side_to_play {
                Side::Attacker => 1,
                Side::Defender => -1,
            },
            final_outcome,
        ));
    }
    training_examples
}

pub fn self_play(nnmodel: CModule, no_games: i32, mcts_iterations: u32, verbose: bool) 
-> Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)>{

    let mut training_data = Vec::new();

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
            let player = game.state.side_to_play;

            if verbose {
                println!("Player: {:?}", player);
                println!("Board:");
                println!("{}", game.state.board);
            }

            let policy = mcts(&nnmodel, &game, 100);
            policy_history.push(policy.clone());

            let mut rng = rng();
            let dist = WeightedIndex::new(&policy).expect("Invalid distribution");
            let action = dist.sample(&mut rng) as u32;
            let str_action: &str = &action_to_str(&action);
            let play = get_ai_play(str_action);
            
            if verbose {
                println!("{}", str_action);
            }

            match game.do_play(play) {
                Ok(status) => {

                    if verbose{
                        println!("Move took: {:?}", move_time.elapsed());
                    }

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
    training_data
}