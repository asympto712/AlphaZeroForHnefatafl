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
    let mut nnmodel = CModule::load_on_device("./agents/test/models/gen0.pt", device).unwrap();

    nnmodel.set_eval();
    (game, nnmodel)
}

fn test_mcts_mcts(num_iter: usize) {
    let (game, nnmodel) = test_setup();
    let start = Instant::now();
    let _ = mcts::mcts(&nnmodel, &game, num_iter);
    let duration = start.elapsed();
    println!("mcts::mcts took {}ms for {} iterations: {}micro secs per iteration", duration.as_millis(), num_iter, duration.as_micros() as f32 / num_iter as f32);
}

fn test_mcts_par_mcts_par(num_iter: usize, num_workers: usize, verbose: bool) {
    let (game, nnmodel) = test_setup();
    let start = Instant::now();

    let nnmodel_ref = Arc::new(nnmodel);
    let mut tree = Tree::new(&game, &nnmodel_ref);
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
}

fn test_mcts_par_mcts_notpar(num_iter: usize, verbose: bool){
    let (game, nnmodel) = test_setup();
    let start = Instant::now();
    let mut tree = Tree::new(&game, &nnmodel);
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
}

fn test_mcts_par_mcts_root_par(num_iter: usize, num_workers: usize) {
    let (game, nnmodel) = test_setup();
    let start = Instant::now();
    let _ = mcts_par::mcts_root_par(Arc::new(nnmodel), &game, num_iter, num_workers);
    let duration = start.elapsed();
    println!("mcts_par::mcts_root_par took {}ms for {} iterations: {}micro secs per iteration", duration.as_millis(), num_iter, duration.as_micros() as f32 / num_iter as f32);
}
fn main() {

    let num_iter = 400;
    let num_workers = 4;
    let verbose = false;
    test_mcts_mcts(num_iter);
    test_mcts_par_mcts_notpar(num_iter, verbose);
    test_mcts_par_mcts_par(num_iter, num_workers, verbose);
    test_mcts_par_mcts_root_par(num_iter, num_workers);
}

