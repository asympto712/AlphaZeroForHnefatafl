use crate::hnefgame::game::state::GameState;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::game::{SmallBasicGame, Game};
use crate::support::{action_to_str, board_to_matrix};
use crate::hnefgame::game::GameOutcome::{Win, Draw};
use crate::hnefgame::game::GameStatus::Over;
use rand::distributions::WeightedIndex;
use tch::nn;
use crate::hnefgame::play::Play;
use rand::prelude::*;
use rand::thread_rng;
use crate::mcts::mcts;
use crate::hnefgame::preset::{boards, rules};
use crate::hnefgame::board::state::BoardState;

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




pub fn self_play(nnmode: CModule, no_games: i32) {

    for i in 0..no_games {

        println!("Game number: {}", i);
    


        let mut game: SmallBasicGame = Game::new(
            rules::KOCH,
            boards::BRANDUBH,
        ).expect("Could not create game.");

        let mut policy_history = Vec::new();

        loop {
            let player = game.state.side_to_play;
            println!("Player: {:?}", player);

            let policy = mcts(nnmodel, game.clone(), 100);
            policy_history.push(policy.clone());

            let mut rng = thread_rng();
            let dist = WeightedIndex::new(&policy).expect("Invalid distribution");
            let play = Play::from_str(&action_to_str(dist.sample(&mut rng)));

            match game.do_play(&play) {
                Ok(status) => {
                    if let Over(outcome) = status {
                        match outcome {
                            Draw(reason) => {
                                println!("Game over. Draw {reason:?}.");
                                let training_examples = generate_training_example(&game.state_history, &policy_history, 0);
                                return;
                            }
                            Win(reason, side) => {
                                let outcome_value = match side {
                                    Side::Attacker => 1,
                                    Side::Defender => -1,
                                };
                                println!("Game over. Winner is {side:?} ({reason:?}).");
                                let training_examples = generate_training_example(&game.state_history, &policy_history, outcome_value);
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Invalid move ({e:?}). Try again.");
                    let max_index = policy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(index, _)| index as u32).expect("Policy vector is empty");
                    let play = Play::from_str(&action_to_str(&max_index));
                }
            }
        }
    }
}