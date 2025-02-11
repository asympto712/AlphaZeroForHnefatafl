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
use egui::{RichText, FontId};
use std::fs::File;
use std::io::Write;
use std::fs;

fn main() {
    let agent_attacker: &str = "300lim_9gen_NoCapReward_0.3Draw/models/gen1.pt";
    let agent_defender: &str = "300lim_9gen_NoCapReward_0.3Draw/models/gen8.pt";

    //Choose from...
    //"mcts_mcts", "mcts_par_mcts_notpar", "mcts_par_mcts_par", "mcts_par_mcts_root_par"
    //  |                   |                       |                   |
    //  -> The one we were using all along for training. Doesn't implement Dirichlet noise for the exploration from the root.
    //                      |                       |                   |
    //                      -> Unparallelized version from mcts_par.rs Uses a bit different data structure. Implements Dirichlet noise.
    //                                              |                   |
    //                                               -> Parallelization ver1 (leaf-parallelization) of mcts_par_mcts_notpar
    //                                                                  |
    //                                                                   -> Parallelization ver2(root parallelization) of mcts_par_mcts_notpar                                 
    let attacker_mcts_alg: &str = "mcts_par_mcts_par";
    let defender_mcts_alg: &str = "mcts_par_mcts_par";

    let attacker_mcts_iter: usize = 100;
    let defender_mcts_iter: usize = 100;
    let no_games: u32 = 1;
    let num_workers: usize = 4;
    let verbose: bool = false; // keep this false

    // Below are the parameters for the MCTS algorithm.
    // c_puct is the exploration constant.
    // It is a hyperparameter that determines how much the MCTS algorithm should explore.

    // alpha is the Dirichlet noise parameter. 
    // It is a hyperparameter that supposedly should reflect the game's action space(?).
    // For the reference, it was kept 0.3 for chess, 0.03 for Go in AlphaZero.

    // eps is the Dirichlet noise parameter.
    // It is a hyperparameter that determines how much noise should be added to the prior probabilities.
    // Should be between (0,1). Higher value means more noise.
    let att_c_puct: f32 = 0.3;
    let att_alpha: f64 = 0.4;
    let att_eps: f32 = 0.4;
    let def_c_puct: f32 = 0.3;
    let def_alpha: f64 = 0.4;
    let def_eps: f32 = 0.4;

    duel_app(agent_attacker,
        agent_defender,
        no_games,
        verbose,
        attacker_mcts_alg,
        attacker_mcts_iter,
        defender_mcts_alg,
        defender_mcts_iter,
        num_workers,
        att_c_puct,
        att_alpha,
        att_eps,
        def_c_puct,
        def_alpha,
        def_eps);    
}

struct MyApp {
    tot_games: u32,
    attacker: String,
    defender: String,
    attacker_mcts_alg: String,
    attacker_mcts_iter: usize,
    att_c_puct: f32,
    att_alpha: f64,
    att_eps: f32,
    defender_mcts_alg: String,
    defender_mcts_iter: usize,
    def_c_puct: f32,
    def_alpha: f64,
    def_eps: f32,
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
            attacker_mcts_iter: 0,
            att_c_puct: 0.0,
            att_alpha: 0.0,
            att_eps: 0.0,
            defender_mcts_alg: "".to_string(),
            defender_mcts_iter: 0,
            def_c_puct: 0.0,
            def_alpha: 0.0,
            def_eps: 0.0,
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

    fn save_results(&self, filename: &str) {
        fs::create_dir_all("duel_log").expect("Unable to create duel_log directory");
        let mut file = File::options().append(true).create(true).open(filename).expect("Unable to open or create file");
        let now = chrono::Local::now();
        writeln!(file, "Date and Time: {}", now.format("%Y-%m-%d %H:%M:%S")).expect("Unable to write data");
        writeln!(file, "Total games: {}", self.tot_games).expect("Unable to write data");
        writeln!(file, "Attacker: {}", self.attacker).expect("Unable to write data");
        writeln!(file, "Defender: {}", self.defender).expect("Unable to write data");
        writeln!(file, "Attacker MCTS algorithm: {}", self.attacker_mcts_alg).expect("Unable to write data");
        writeln!(file, "Attacker MCTS iterations: {}, c_puct: {}, alpha: {}, eps: {}", self.attacker_mcts_iter, self.att_c_puct, self.att_alpha, self.att_eps).expect("Unable to write data");
        writeln!(file, "Defender MCTS algorithm: {}", self.defender_mcts_alg).expect("Unable to write data");
        writeln!(file, "Defender MCTS iterations: {}, c_puct: {}, alpha: {}, eps: {}", self.defender_mcts_iter, self.def_c_puct, self.def_alpha, self.def_eps).expect("Unable to write data");
        writeln!(file, "Game number: {}", self.game_num).expect("Unable to write data");
        writeln!(file, "Attacker wins: {}", self.winsatt).expect("Unable to write data");
        writeln!(file, "Defender wins: {}", self.winsdef).expect("Unable to write data");
        writeln!(file, "Draws: {}", self.draws).expect("Unable to write data");
        writeln!(file, "Minimum game length: {}", self.minlength).expect("Unable to write data");
        writeln!(file, "Maximum game length: {}", self.maxlength).expect("Unable to write data");
        writeln!(file, "Average game length: {}", self.avelength).expect("Unable to write data");
        writeln!(file, "Attacker capture count: {}", self.attacker_capture_count).expect("Unable to write data");
        writeln!(file, "Defender capture count: {}", self.defender_capture_count).expect("Unable to write data");
        writeln!(file, "\n").expect("unable to write data");
    }
}

struct AppWrapper {
    app: Arc<Mutex<MyApp>>,
}

impl eframe::App for AppWrapper{
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let app = self.app.lock().unwrap();
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("Save Results").clicked() {
                app.save_results("duel_log/results.txt");
            }
            ui.label("You can push this button to save the result to duel_log/results.txt.");

            ui.separator();
            ui.heading("metadata");
            ui.label(format!("Total games: {}", app.tot_games));
            ui.horizontal(|ui| {
                ui.label("Attacker: ");
                ui.label(format!("{}", app.attacker));
            });
            ui.horizontal(|ui|{
                ui.label(format!("MCTS Alg: {}", app.attacker_mcts_alg));
                ui.label(format!("Iterations: {}", app.attacker_mcts_iter));
                ui.label(format!("c_puct: {}", app.att_c_puct));
                ui.label(format!("alpha: {}", app.att_alpha));
                ui.label(format!("eps: {}", app.att_eps));
            });
            ui.horizontal(|ui| {
                ui.label("Defender: ");
                ui.label(format!("{}", app.defender));
            });
            ui.horizontal(|ui|{
                ui.label(format!("MCTS Alg: {}", app.defender_mcts_alg));
                ui.label(format!("Iterations: {}", app.defender_mcts_iter));
                ui.label(format!("c_puct: {}", app.def_c_puct));
                ui.label(format!("alpha: {}", app.def_alpha));
                ui.label(format!("eps: {}", app.def_eps));
            });
            ui.separator();
            ui.heading("Game Results");
            ui.horizontal(|ui|{
                ui.label(format!("Game number: {}", app.game_num));
                ui.label(format!("Attacker wins: {}", app.winsatt));
                ui.label(format!("Defender wins: {}", app.winsdef));
                ui.label(format!("Draws: {}", app.draws));
            });
            ui.separator();
            ui.heading("Game Stats");
            ui.horizontal(|ui|{
                ui.label(format!("Minimum length: {}", app.minlength));
                ui.label(format!("Maximum length: {}", app.maxlength));
                ui.label(format!("Average length: {}", app.avelength));
            });
            ui.horizontal(|ui|{
                ui.label(format!("Attacker capture count: {}", app.attacker_capture_count));
                ui.label(format!("Defender capture count: {}", app.defender_capture_count));
            });
            ui.separator();
            ui.heading("Current Game State");
            ui.label(format!("Move time: {}ms", app.move_time));
            ui.label(format!("Current turn: {}", app.current_turn));
            ui.centered_and_justified(|ui| {
                ui.monospace(RichText::new(format!("Board: \n{}", app.board)).font(FontId::proportional(20.0)));
            });
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
    num_workers: usize,
    c_puct: f32,
    alpha: f64,
    eps: f32) 
    -> Vec<f32>{

    match mcts_alg {
        MCTSAlg::MctsMcts => {
            let policy = mcts::mcts(&nnmodel, game, num_iter);
            policy
        },
        MCTSAlg::MctsParMctsNotpar => {
            let policy = mcts_par::mcts_notpar(&nnmodel, game, num_iter, c_puct, alpha, eps);
            policy
        },
        MCTSAlg::MctsParMctsPar => {
            let policy = mcts_par::mcts_par(nnmodel, game, num_iter, num_workers, c_puct, alpha, eps);
            policy
        },
        MCTSAlg::MctsParMctsRootPar => {
            let policy = mcts_par::mcts_root_par(nnmodel, game, num_iter, num_workers, c_puct, alpha, eps);
            policy
        },
    }
}

fn duel(agent_attacker: &str,
        agent_defender: &str,
        no_games: u32,
        verbose: bool,
        attacker_mcts_alg: &str,
        attacker_mcts_iter: usize,
        defender_mcts_alg: &str,
        defender_mcts_iter: usize,
        num_workers: usize,
        att_c_puct: f32,
        att_alpha: f64,
        att_eps: f32,
        def_c_puct: f32,
        def_alpha: f64,
        def_eps: f32,
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
                    let policy: Vec<f32> = mcts_do_alg(
                        &attacker_mctsalg,
                        Arc::clone(&nnmodel_attacker),
                        &game,
                        attacker_mcts_iter,
                        num_workers,
                        att_c_puct,
                        att_alpha,
                        att_eps);

                    // println!("{}", policy.len());
                    // let sum: f32 = policy.iter().sum();
                    // println!("Sum of policy values: {}", sum);

                    let action: u32 = 
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
                    let policy: Vec<f32> = mcts_do_alg(
                        &defender_mctsalg,
                        Arc::clone(&nnmodel_defender),
                        &game,
                        defender_mcts_iter,
                        num_workers,
                        def_c_puct,
                        def_alpha,
                        def_eps);

                    let action: u32 = 
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

fn duel_app(
    agent_attacker: &str,
    agent_defender: &str,
    no_games: u32,
    verbose: bool,
    attacker_mcts_alg: &str,
    attacker_mcts_iter: usize,
    defender_mcts_alg: &str,
    defender_mcts_iter: usize,
    num_workers: usize,
    att_c_puct: f32,
    att_alpha: f64,
    att_eps: f32,
    def_c_puct: f32,
    def_alpha: f64,
    def_eps: f32) 
    {
    
    let attacker_path = format!("agents/{}", agent_attacker);
    let defender_path = format!("agents/{}", agent_defender);

    let (tx, rx) = mpsc::channel();
    let app = Arc::new(Mutex::new(MyApp::default()));
    
    // The settings for the app. It should match the arguments we give to func duel.
    // Change if necessary
    {
        let mut app = app.lock().unwrap();
        app.tot_games = no_games;
        app.attacker = agent_attacker.to_string();
        app.defender = agent_defender.to_string();
        app.attacker_mcts_alg = attacker_mcts_alg.to_string();
        app.attacker_mcts_iter = attacker_mcts_iter;
        app.att_alpha = att_alpha;
        app.att_c_puct = att_c_puct;
        app.att_eps = att_eps;
        app.defender_mcts_alg = defender_mcts_alg.to_string();
        app.defender_mcts_iter = defender_mcts_iter;
        app.def_alpha = def_alpha;
        app.def_c_puct = def_c_puct;
        app.def_eps = def_eps;
    }
    let app_clone = Arc::clone(&app);

    let attacker_mcts_alg_clone = attacker_mcts_alg.to_string();
    let defender_mcts_alg_clone = defender_mcts_alg.to_string();
    std::thread::spawn(move || {
        duel(
            &attacker_path,
            &defender_path,
            no_games,
            verbose,
            &attacker_mcts_alg_clone,
            attacker_mcts_iter,
            &defender_mcts_alg_clone,
            defender_mcts_iter,
            num_workers,
            att_c_puct,
            att_alpha,
            att_eps,
            def_c_puct,
            def_alpha,
            def_eps,
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

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 700.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "duel results recorder",
        options,
        Box::new(|_cc| Ok(Box::new(AppWrapper{app: Arc::clone(&app) }))),
    ).unwrap();
}