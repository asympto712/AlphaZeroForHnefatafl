use rand::prelude::*;
use hnefgame::game::{GameState, Side, SmallBasicGame, Game};

fn generate_training_example(
    game_state_history: Vec<GameState>,
    policy_history: Vec<Vec<f32>>,
    final_outcome: i32,
) -> Vec<(GameState, Vec<f32>, i32, i32)> {
    let mut training_examples = Vec::new();

    for (state, policy) in game_state_history.iter().zip(policy_history.iter()) {
        training_examples.push((
            state.clone(),
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

pub fn self_play() {
    let mut game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");

    let mut policy_history = Vec::new();

    loop {
        let player = game.state.side_to_play;
        println!("Player: {:?}", player);


        let policy = mcts(nnmodel, game.clone(), 100);
        policy_history.push(policy.clone());

        let mut rng = thread_rng();
        let dist = rand::distributions::WeightedIndex::new(&policy).expect("Invalid distribution");
        let play = action_to_str(dist.sample(&mut rng));

        match game.do_play(play) {
            Ok(status) => {
                if let Over(outcome) = status {
                    match outcome {
                        Draw(reason) => {
                            println!("Game over. Draw {reason:?}.");
                            let training_examples = generate_training_example(game.state_history, policy_history, 0);
                            return;
                        }
                        Win(reason, side) => {
                            let outcome_value = match side {
                                Side::Attacker => 1,
                                Side::Defender => -1,
                            };
                            println!("Game over. Winner is {side:?} ({reason:?}).");
                            let training_examples = generate_training_example(game.state_history, policy_history, outcome_value);
                            return;
                        }
                    }
                }
            }
            Err(e) => {
                println!("Invalid move ({e:?}). Try again.");
                let play = action_to_str(policy.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(index, _)| index).expect("Policy vector is empty"));
                game.do_play(play).expect("Failed to play the best move");
            }
        }
    }
}