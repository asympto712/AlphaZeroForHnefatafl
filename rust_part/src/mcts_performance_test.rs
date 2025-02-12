#![allow(unused_imports)]
use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
use crate::hnefgame::board::state::BoardState;
use crate::hnefgame::board::state::BitfieldBoardState;
use tch::CModule;
mod hnefgame;
mod mcts;
mod support;
mod mcts_par;
use crate::mcts::mcts;
use crate::mcts_par::{Tree, Node};
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;
use std::env;
use std::ffi::CString;
use std::os::raw::c_char;
use winapi::um::libloaderapi::LoadLibraryA;
use std::fs::OpenOptions;
use std::io::Write;


fn main() {

    /* 
    let path = CString::new("Path/to/lib/torch_cuda.dll").unwrap();
    
    unsafe {
        LoadLibraryA(path.as_ptr() as *const c_char);
    }
    */

    // let num_iter: usize = 100;
    // let num_workers: usize = 4;
    // let verbose: bool = false;
    // let c_puct: f32 = 0.3;
    // let alpha: f64 = 0.4;
    // let eps: f32 = 0.25;

    // let _ = test_mcts_mcts(num_iter);
    // let _ = test_mcts_par_mcts_notpar(num_iter, verbose, c_puct, alpha, eps);
    // let _ = test_mcts_par_mcts_par(num_iter, num_workers, verbose, c_puct, alpha, eps);
    // let _ = test_mcts_par_mcts_root_par(num_iter, num_workers, c_puct, alpha, eps);
    test_log_stats();
}

fn test_log_stats(){
    let verbose = false;
    let c_puct: f32 = 0.3;
    let alpha: f64 = 0.4;
    let eps: f32 = 0.25;
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("mcts_performance.csv")
        .expect("Unable to open or create file");
    let msg = "Unable to write to file";
    let num_iters: [usize; 3] = [100, 400, 800];
    let num_workers: [usize; 2] = [4,8];
    for num_iter in num_iters.iter(){
        for num_worker in num_workers.iter(){
            writeln!(file, "num_iter,{},num_worker,{}", num_iter, num_worker).expect(msg);
            writeln!(file, "1st mcts,mcts_notpar,mcts_par,mcts_rootpar").expect(msg);
            for _ in 0..20{
                let dur1 = test_mcts_mcts(*num_iter);
                let dur2 = test_mcts_par_mcts_notpar(*num_iter, verbose, c_puct, alpha, eps);
                let dur3 = test_mcts_par_mcts_par(*num_iter, *num_worker, verbose, c_puct, alpha, eps);
                let dur4 = test_mcts_par_mcts_root_par(*num_iter, *num_worker, c_puct, alpha, eps);
                writeln!(file, "{},{},{},{}", dur1, dur2, dur3, dur4).expect(msg);
            }
        }
    }
    file.flush().expect("Unable to flush file");
}

fn test_setup() -> (Game<BitfieldBoardState<u64>>, CModule){
    let game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");

    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };

    if device == tch::Device::Cpu {println!("nnmodel was loaded onto CPU");}
    else {println!("nnmodel was loaded onto CUDA");}

    // Replace the agent path with the path to the model you want to test
    let mut nnmodel = CModule::load_on_device("./agents/300lim_4gen_plain_vanilla/models/gen4.pt", device).unwrap();

    nnmodel.set_eval();
    (game, nnmodel)
}

fn test_mcts_mcts(num_iter: usize) -> u128 {
    let (game, nnmodel) = test_setup();
    let start = Instant::now();
    let _ = mcts::mcts(&nnmodel, &game, num_iter);
    let duration = start.elapsed();
    println!("mcts::mcts took {}ms for {} iterations: {}micro secs per iteration", duration.as_millis(), num_iter, duration.as_micros() as f32 / num_iter as f32);
    duration.as_millis()
}

fn test_mcts_par_mcts_par(num_iter: usize, num_workers: usize, verbose: bool, c_puct: f32, alpha: f64, eps: f32) -> u128 {
    let (game, nnmodel) = test_setup();
    let start = Instant::now();

    let nnmodel_ref = Arc::new(nnmodel);
    let mut tree = Tree::new(&game, &nnmodel_ref, c_puct, alpha, eps);
    let _result = tree.mcts_par(Arc::clone(&nnmodel_ref), num_iter, num_workers);

    let duration = start.elapsed();
    if verbose{
        for i in 0..1{
            let node = tree.refs[i].borrow();
            if let Node::Notr(notr) = &*node{
                println!("Info of node number {}", i);
                notr.display_info();
            }
        }
        println!("Length of tree.refs: {}", tree.refs.len());
        let mut depth_vec = Vec::new();
        for node_ref in &tree.refs {
            let node = node_ref.borrow();
            if let Node::Notr(notr) = &*node {
                depth_vec.push(notr.depth);
            } 
        }
        let max_depth = depth_vec.iter().max().unwrap_or(&0);
        println!("Maximum depth: {}", max_depth);
    }

    println!("mcts_par::mcts_par took {}ms for {} iterations: {}micro secs per second", duration.as_millis(), num_iter, duration.as_micros() as f32 / num_iter as f32);
    duration.as_millis()
}

fn test_mcts_par_mcts_notpar(num_iter: usize, verbose: bool, c_puct: f32, alpha: f64, eps: f32) -> u128{
    let (game, nnmodel) = test_setup();
    let start = Instant::now();
    let mut tree = Tree::new(&game, &nnmodel, c_puct, alpha, eps);
    let _ = tree.mcts_notpar(&nnmodel, num_iter);
    let duration = start.elapsed();

    if verbose{
        for i in 0..1{
            let node = tree.refs[i].borrow();
            if let Node::Notr(notr) = &*node{
                println!("Info of node number {}", i);
                notr.display_info();
            }
        }
        println!("Length of tree.refs: {}", tree.refs.len());
        let mut depth_vec = Vec::new();
        for node_ref in &tree.refs {
            let node = node_ref.borrow();
            if let Node::Notr(notr) = &*node {
                depth_vec.push(notr.depth);
            } 
        }
        let max_depth = depth_vec.iter().max().unwrap_or(&0);
        println!("Maximum depth: {}", max_depth);
    }
    println!("mcts_par::mcts_notpar took {}ms for {} iterations: {}micro secs per iteration", duration.as_millis(), num_iter, duration.as_micros() as f32 / num_iter as f32);
    duration.as_millis()
}

fn test_mcts_par_mcts_root_par(num_iter: usize, num_workers: usize, c_puct: f32, alpha: f64, eps: f32)-> u128 {
    let (game, nnmodel) = test_setup();
    let start = Instant::now();
    let _ = mcts_par::mcts_root_par(Arc::new(nnmodel), &game, num_iter, num_workers, c_puct, alpha, eps);
    let duration = start.elapsed();
    println!("mcts_par::mcts_root_par took {}ms for {} iterations: {}micro secs per iteration", duration.as_millis(), num_iter, duration.as_micros() as f32 / num_iter as f32);
    duration.as_millis()
}

