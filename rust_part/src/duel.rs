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
use crate::hnefgame::board::state::{BoardState, BitfieldBoardState};
use crate::support::{action_to_str, get_ai_play};
use crate::mcts::mcts;
use crate::mcts_par::{Tree, Node};

use tch::CModule;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Sender};
use eframe::egui;

struct MyApp {
    tot_games: u32,
    attacker: String,
    defender: String,
    attacker_mcts_alg: String,
    defender_mcts_alg: String,
    game_num: u32,
    winsatt: u32,
    winsdef: u32,
    draws: u32,
    minlength: u32,
    maxlength: u32,
    avelength: f32,
    move_time: f32,
    current_turn: u32,
    attacker_capture_count: u32,
    defender_capture_count: u32,
    board: BitfieldBoardState<u64>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            tot_games: 0,
            attacker: "".to_string(),
            defender: "".to_string(),
            attacker_mcts_alg: "".to_string(),
            defender_mcts_alg: "".to_string(),
            game_num: 0,
            winsatt: 0,
            winsdef: 0,
            draws: 0,
            minlength: 0,
            maxlength: 0,
            avelength: 0.0,
            move_time: 0.0,
            current_turn: 0,
            attacker_capture_count: 0,
            defender_capture_count: 0,
            board: BitfieldBoardState::default(),
        }
    }
}

impl MyApp {
    fn update_results(
        &mut self,
        game_num: u32,
        winsatt: u32,
        winsdef: u32,
        draws: u32,
        minlength: u32,
        maxlength: u32,
        avelength: f32,
        move_time: f32,
        current_turn: u32,
        attacker_capture_count: u32,
        defender_capture_count: u32,
        board: BitfieldBoardState<u64>
    ){
        self.game_num = game_num;
        self.winsatt = winsatt;
        self.winsdef = winsdef;
        self.draws = draws;
        self.minlength = minlength;
        self.maxlength = maxlength;
        self.avelength = avelength;
        self.move_time = move_time;
        self.current_turn = current_turn;
        self.attacker_capture_count = attacker_capture_count;
        self.defender_capture_count = defender_capture_count;
        self.board = board;
    }
}

struct AppWrapper {
    app: Arc<Mutex<MyApp>>,
}

impl eframe::App for AppWrapper{
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let app = self.app.lock().unwrap();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("metadata");
            ui.label(format!("Total games: {}", app.tot_games));
            ui.label(format!("Attacker: {}", app.attacker));
            ui.label(format!("Defender: {}", app.defender));
            ui.label(format!("Attacker MCTS algorithm: {}", app.attacker_mcts_alg));
            ui.label(format!("Defender MCTS algorithm: {}", app.defender_mcts_alg));
            ui.heading("Game Results");
            ui.label(format!("Game number: {}", app.game_num));
            ui.label(format!("Attacker wins: {}", app.winsatt));
            ui.label(format!("Defender wins: {}", app.winsdef));
            ui.label(format!("Draws: {}", app.draws));
            ui.label(format!("Minimum game length: {}", app.minlength));
            ui.label(format!("Maximum game length: {}", app.maxlength));
            ui.label(format!("Average game length: {}", app.avelength));
            ui.label(format!("Move time: {}ms", app.move_time));
            ui.label(format!("Current turn: {}", app.current_turn));
            ui.label(format!("Attacker capture count: {}", app.attacker_capture_count));
            ui.label(format!("Defender capture count: {}", app.defender_capture_count));
            ui.label(format!("Board: \n{}", app.board));
            ui.ctx().request_repaint();
        });
    }
}

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
        num_workers: usize,
        tx: Sender<(u32, u32, u32, u32, u32, u32, f32, f32, u32, u32, u32, BitfieldBoardState<u64>)>) {

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

    let mut game_num: u32 = 0;
    let mut wins1: u32 = 0;
    let mut wins2: u32 = 0;
    let mut draws: u32 = 0;
    let mut game_length: Vec<u32> = Vec::new();
    let mut ave_game_length: f32 = 0.0;
    let mut move_time: f32 = 0.0;
    let mut attacker_capture_count: u32 = 0;
    let mut defender_capture_count: u32 = 0;

    for _ in 0..no_games {
        let mut game = SmallBasicGame::new(rules::KOCH, boards::BRANDUBH).unwrap();
        let mut count: u32 = 0;
        game_num += 1;

        loop {

            let tx = tx.clone();

            let min_length = game_length.iter().min().unwrap_or(&0);
            let max_length = game_length.iter().max().unwrap_or(&0);

            tx.send((game_num, wins1, wins2, draws, *min_length, *max_length, ave_game_length, move_time, count, attacker_capture_count, defender_capture_count, game.state.board)).unwrap();

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
            move_time = duration.as_millis() as f32;

            if verbose{
                println!("Move took: {:?}\n", duration);
            }

            match game.do_play(play){

                Ok(status) => {

                    count += 1;
                    let effects = &game.play_history.last().unwrap().effects;
                    let captures = effects.captures.len();
                    match player {
                        Side::Attacker => {
                            attacker_capture_count += captures as u32;
                        },
                        Side::Defender => {
                            defender_capture_count += captures as u32;
                        },
                    }
            
                    if let Over(outcome) = status {
                        match outcome {
                            Draw(reason) => {
                                println!("Game over. Draw {reason:?}.");
                                draws += 1;

                                tx.send((
                                    game_num,
                                    wins1,
                                    wins2,
                                    draws,
                                    *min_length,
                                    *max_length,
                                    ave_game_length,
                                    move_time,
                                    count,
                                    attacker_capture_count,
                                    defender_capture_count,
                                    game.state.board)).unwrap();
                                break;
                            }
                            Win(reason, side) => {
                                println!("Game over. Winner is {side:?} ({reason:?}).");
                                if side == Side::Attacker {
                                    wins1 += 1;
                                } else {
                                    wins2 += 1;
                                }
                                tx.send((
                                    game_num,
                                    wins1,
                                    wins2,
                                    draws,
                                    *min_length,
                                    *max_length,
                                    ave_game_length,
                                    move_time,
                                    count,
                                    attacker_capture_count,
                                    defender_capture_count,
                                    game.state.board)).unwrap();
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
        game_length.push(count);
        ave_game_length = game_length.iter().sum::<u32>() as f32 / game_length.len() as f32;
    }
    println!("total games: {}", game_num);
    println!("Agent 1 wins: {}", wins1);
    println!("Agent 2 wins: {}", wins2);
    println!("Draws: {}", draws);
    println!("Winrate of Agent 1: {}", wins1 as f32 / no_games as f32);
    println!("Winrate of Agent 2: {}", wins2 as f32 / no_games as f32);
    println!("Draw rate: {}", draws as f32 / no_games as f32);
    println!("Minimum game length: {}", game_length.iter().min().unwrap());
    println!("Maximum game length: {}", game_length.iter().max().unwrap());
    println!("Average game length: {}", ave_game_length);
    println!("Attacker has captured: {} in total", attacker_capture_count);
    println!("Defender has captured: {} in total", defender_capture_count);
}

fn main() {
    let agent_attacker = "300lim_9gen_NoCapReward_0.3Draw/models/gen1.pt";
    let agent_defender = "300lim_9gen_NoCapReward_0.3Draw/models/gen8.pt";
    let attacker_path = format!("agents/{}", agent_attacker);
    let defender_path = format!("agents/{}", agent_defender);

    let (tx, rx) = mpsc::channel();
    let app = Arc::new(Mutex::new(MyApp::default()));
    
    // The settings for the app. It should match the arguments we give to func duel.
    // Change if necessary
    {
        let mut app = app.lock().unwrap();
        app.tot_games = 10;
        app.attacker = agent_attacker.to_string();
        app.defender = agent_defender.to_string();
        app.attacker_mcts_alg = "mcts_par_mcts_par".to_string();
        app.defender_mcts_alg = "mcts_par_mcts_par".to_string();
    }
    let app_clone = Arc::clone(&app);

    std::thread::spawn(move || {
        duel(
            &attacker_path,
            &defender_path,
            10,
            100,
            false,
            "mcts_par_mcts_par",
            "mcts_par_mcts_par",
            4,
            tx);
    });

    std::thread::spawn(move || {
        loop {
            if let Ok((
                game_num,
                wins1,
                wins2,
                draws,
                min_length,
                max_length,
                ave_game_length,
                move_time,
                current_turn,
                attacker_capture_count,
                defender_capture_count,
                board)) = rx.recv() {
                // Debugging
                // println!("Received results");
                // println!("Attacker wins: {}", wins1);
                let mut app = app_clone.lock().unwrap();
                app.update_results(
                    game_num, 
                    wins1,
                    wins2,
                    draws,
                    min_length,
                    max_length,
                    ave_game_length,
                    move_time,
                    current_turn,
                    attacker_capture_count,
                    defender_capture_count,
                    board);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    });

    let native_options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "duel results recorder",
        native_options,
        Box::new(|_cc| Ok(Box::new(AppWrapper{app: Arc::clone(&app) }))),
    ).unwrap();
}