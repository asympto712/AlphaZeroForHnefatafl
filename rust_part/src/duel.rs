#![allow(unused_imports)]
mod support;
mod mcts;
mod hnefgame;
mod mcts_par;

use crate::hnefgame::game::state::GameState;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Win, Draw};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
use crate::hnefgame::preset::{boards, rules};
use crate::hnefgame::board::state::BoardState;
use crate::support::{action_to_str, get_ai_play};
use crate::mcts::mcts;
use crate::mcts_par::{Tree, Node};

use tch::CModule;
use std::time::Instant;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
enum MCTSAlg {
    MctsMcts,
    MctsParMctsNotpar,
    MctsParMctsPar,
    MctsParMctsRootPar,
}

impl std::str::FromStr for MCTSAlg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mcts_mcts" => Ok(MCTSAlg::MctsMcts),
            "mcts_par_mcts_notpar" => Ok(MCTSAlg::MctsParMctsNotpar),
            "mcts_par_mcts_par" => Ok(MCTSAlg::MctsParMctsPar),
            "mcts_par_mcts_root_par" => Ok(MCTSAlg::MctsParMctsRootPar),
            _ => Err("Invalid MCTS algorithm".to_string()),
        }
    }
}

fn mcts_do_alg<T: BoardState + Send + 'static>(
    mcts_alg: &MCTSAlg,
    nnmodel: Arc<CModule>,
    game: &Game<T>,
    num_iter:usize,
    num_workers: usize) 
    -> Vec<f32>{

    match mcts_alg {
        MCTSAlg::MctsMcts => {
            let policy = mcts::mcts(&nnmodel, game, num_iter);
            policy
        },
        MCTSAlg::MctsParMctsNotpar => {
            let policy = mcts_par::mcts_notpar(&nnmodel, game, num_iter);
            policy
        },
        MCTSAlg::MctsParMctsPar => {
            let policy = mcts_par::mcts_par(nnmodel, game, num_iter, num_workers);
            policy
        },
        MCTSAlg::MctsParMctsRootPar => {
            let policy = mcts_par::mcts_root_par(nnmodel, game, num_iter, num_workers);
            policy
        },
    }
}

fn duel(agent_attacker: &str,
        agent_defender: &str,
        no_games: u32,
        mcts_iterations: usize,
        verbose: bool,
        attacker_mcts_alg: &str,
        defender_mcts_alg: &str,
        num_workers: usize) {

    let attacker_mctsalg = attacker_mcts_alg.parse::<MCTSAlg>().unwrap();
    let defender_mctsalg = defender_mcts_alg.parse::<MCTSAlg>().unwrap();

    let mut nnmodel_attacker = 
    if tch::Cuda::is_available() {
        CModule::load_on_device(agent_attacker, tch::Device::Cuda(0)).unwrap()
    } else {
        CModule::load_on_device(agent_attacker, tch::Device::Cpu).unwrap()
    };
    nnmodel_attacker.set_eval();
    let nnmodel_attacker = Arc::new(nnmodel_attacker);

    let mut nnmodel_defender = 
    if tch::Cuda::is_available() {
        CModule::load_on_device(agent_defender, tch::Device::Cuda(0)).unwrap()
    } else {
        CModule::load_on_device(agent_defender, tch::Device::Cpu).unwrap()
    };
    nnmodel_defender.set_eval();
    let nnmodel_defender = Arc::new(nnmodel_defender);

    let mut wins1 = 0;
    let mut wins2 = 0;
    let mut draws = 0;

    for _ in 0..no_games {
        let mut game = SmallBasicGame::new(rules::KOCH, boards::BRANDUBH).unwrap();

        loop {
            let player = game.state.side_to_play;
            if verbose {
                println!("Player: {:?}", player);
                println!("Board:");
                println!("{}", game.state.board);
            }
            let start = Instant::now();
            let play: Play = match player {
                Side::Attacker => {
                    let policy = mcts_do_alg(&attacker_mctsalg, Arc::clone(&nnmodel_attacker), &game, mcts_iterations, num_workers);

                    // println!("{}", policy.len());
                    // let sum: f32 = policy.iter().sum();
                    // println!("Sum of policy values: {}", sum);

                    let action = 
                    policy.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(index, _)| index)
                    .unwrap() as u32;

                    let str_action: &str = &action_to_str(&action);

                    if verbose{
                        println!("{}", str_action);
                    }

                    get_ai_play(str_action)
                    
                },
                Side::Defender => {
                    let policy = mcts_do_alg(&defender_mctsalg, Arc::clone(&nnmodel_defender), &game, mcts_iterations, num_workers);

                    // Debugging
                    // println!("{}", policy.len());
                    // let sum: f32 = policy.iter().sum();
                    // println!("Sum of policy values: {}", sum);
                    // for (index, value) in policy.iter().enumerate() {
                    //     if *value != 0.0 {
                    //         println!("Index: {}, Value: {}", index, value);
                    //     }
                    // }

                    let action = 
                    policy.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(index, _)| index)
                    .unwrap() as u32;

                    let str_action: &str = &action_to_str(&action);

                    if verbose{
                        println!("{}", str_action);
                    }

                    get_ai_play(str_action)
                },
            };

            let duration = start.elapsed();

            if verbose{
                println!("Move took: {:?}\n", duration);
            }

            match game.do_play(play){

                Ok(status) => {
            
                    if let Over(outcome) = status {
                        match outcome {
                            Draw(reason) => {
                                println!("Game over. Draw {reason:?}.");
                                draws += 1;
                                break;
                            }
                            Win(reason, side) => {
                                println!("Game over. Winner is {side:?} ({reason:?}).");
                                if side == Side::Attacker {
                                    wins1 += 1;
                                } else {
                                    wins2 += 1;
                                }
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Invalid move ({e:?}). Try again.");
                    continue
                }
            }     
        }
    }
    println!("Agent 1 wins: {}", wins1);
    println!("Agent 2 wins: {}", wins2);
    println!("Draws: {}", draws);
    println!("Winrate of Agent 1: {}", wins1 as f32 / no_games as f32);
    println!("Winrate of Agent 2: {}", wins2 as f32 / no_games as f32);
}

fn main(){
    let agent_attacker = "300lim_9gen_NoCapReward_0.3Draw/models/gen1.pt";
    let agent_defender = "300lim_9gen_NoCapReward_0.3Draw/models/gen8.pt";
    let attacker_path = format!("agents/{}", agent_attacker);
    let defender_path = format!("agents/{}", agent_defender);
    duel(&attacker_path, &defender_path, 10, 100, true, "mcts_par_mcts_par", "mcts_par_mcts_par", 4);
}