#![allow(unused_imports)]
use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::play::Play;
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


fn main() {

    let start = Instant::now();
    let game: SmallBasicGame = Game::new(
        hnefgame::preset::rules::KOCH,
        hnefgame::preset::boards::BRANDUBH,
    ).expect("Could not create game.");

    // let play: Play = support::get_ai_play("d1-e1");
    // println!("{play:?}");

    // println!("{} to {}", 2252, support::action_to_str(&2252));

    let mut nnmodel = CModule::load("./agents/test/models/gen0.pt").unwrap();
    nnmodel.set_eval();

    let nnmodel_ref = Arc::new(nnmodel);
    let mut tree = Tree::new(&game, Arc::clone(&nnmodel_ref));
    let _result = tree.mcts_par(Arc::clone(&nnmodel_ref), 100);

    for i in 0..10{
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
    let duration = start.elapsed();
    println!("{}ms", duration.as_millis());
}

