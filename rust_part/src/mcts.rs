#![allow(non_snake_case)]
#![allow(unused_imports)]

use crate::hnefgame::game::{Game, SmallBasicGame};
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::Over;
use crate::hnefgame::game::state::GameState;
use crate::hnefgame::play::Play;
use crate::support::{board_to_matrix, get_indices_of_ones, generate_tile_plays, get_play, action_to_str};
use crate::hnefgame::pieces::Side;
use crate::hnefgame::board::state::BoardState;
use std::any::type_name;
use std::collections::HashMap;
use rand::prelude::*;
use tch::{CModule, Tensor, Kind, Device};

type Action = u32;
type Board = Vec<Vec<u32>>;  // Here we assume the Board is already converted into matrix representation & flattened. Update: nvm

const C_PUCT: f32 = 0.3;

#[derive(Debug, Clone)]
struct Node<'a> {
    board: Board, 
    parent: Option<(&'a Node<'a>, Action)>,
    children: HashMap<Action, Box<&'a Node<'a>>>, 
    visits: f32,
    valid_actions: Vec<Action>, 
    action_probs: HashMap<Action, f32>,
    action_counts: HashMap<Action, f32>, 
    action_Qs: HashMap<Action, f32>,
}

impl Node<'_>{
    fn new(board: Board,
           parent: Option<(&Node, Action)>,
           valid_actions: Vec<Action>,
           visits: f32) -> Self {
        Node {
            board,
            parent,
            children: HashMap::new(),
            visits,
            valid_actions,
            action_probs: HashMap::new(),
            action_counts: HashMap::new(),
            action_Qs: HashMap::new(),
        }
    }

    fn uct_value(&self, action: &Action) -> f32 {
        // if self.action_counts(action) == 0.0 {
        //     return f64::INFINITY;
        // }
        // To do; add some dirichlet noise
        return self.action_Qs(action) + C_PUCT * self.action_probs(action) * self.visits.sqrt() / (1 + self.action_counts(action));
    }

}

// Current one.
fn search<T: BoardState>(game_state: &mut GameState<T>, node: &mut Node, nnmodel: &CModule) -> f32 {

    let current_player = game_state.side_to_play;

    match game_state.status {
        Ongoing => (),
        Over(outcome) => match outcome {
            Win(_, side) => {
                if current_player == side {
                    let reward = 1.0;
                    return -1.0 * reward;
                } else {
                    let reward = -1.0;
                    return -1.0 * reward;
                }
            }
            Draw(_) => return 0.0,
        },
    }


    // TODO: add a logic for when there IS no valid_actions
    // TEMPORARY SOLUTION

    if node.valid_actions.is_empty() {
        let reward = -1.0;
        return -1.0 * reward;
    }
    
    let action = node.valid_actions
            .iter()
            .max_by(|a, b| {
                node.uct_value(a).partial_cmp(&node.uct_value(b)).unwrap()
            })
            .unwrap();
    
    let play_string = action_to_str(action);
    let play: Play = Play::from_str(&play_string);

    game_state.do_play(play); //check this again
        
    if !node.children.contains_key(action) {         
        let reward = expand(&mut node, &action, &game_state, &nnmodel);
        return -1.0 * reward
    }

    let next_node = **node.children.get(action).unwrap();
    let reward = search(game_state, &mut next_node, &nnmodel);

    node.actions_Qs(action) = (node.actions_counts(action) * node.actions_Qs(action) + reward) / (node.actions_counts(action) + 1.0);
    node.actions_counts(action) += 1.0;
    node.visits += 1.0;
    return -1 * reward
    
}

fn expand<T: BoardState>(parent: &mut Node, action: &Action, game_state: &GameState<T>, nnmodel: &CModule) {
    let (valid_actions, pi, value) = model_predict(game_state, nnmodel);
    let num_valid_actions = valid_actions.len();
    let mut action_counts = HashMap::with_capacity(num_valid_actions);
    let mut action_Qs = HashMap::with_capacity(num_valid_actions);
    let mut action_probs = HashMap::with_capacity(num_valid_actions);

    if !valid_actions.is_empty() {
        for action in valid_actions {
            action_counts.insert(*action, 0.0);
            action_Qs.insert(*action, 0.0);
            action_probs.insert(*action, pi[action.try_into().expect("could not convert action into an integer")]);
        }
    }

    let mut new_node: Node = Node{
        board: board_to_matrix(&game_state.board), //check ownership
        parent: (parent, action),
        children: HashMap::new(),
        visits: 1.0,
        valid_actions,
        action_probs,
        action_counts,
        action_Qs,
    };
    parent.children.insert(action, Box::new(&new_node));

    return value
}

fn model_predict<T: BoardState>(game_state: &GameState<T>, nnmodel: &CModule) -> (Vec<f32>, Vec<f32>, f32) {

    // Preparing input Tensors
    let matrix_representation = board_to_matrix(game_state);

    let player = match game_state.side_to_play {
        Side::Attacker => 1,
        Side::Defender => -1,
    };

    let board = Tensor::from_slice2(&matrix_representation); //Assuming game.state is a 2-dimensional vector
    let cond: [bool; 1] = if player == 1 {[true]} else {[false]};
    let cond: Tensor = Tensor::from_slice(&cond);

    let (prob, value): (Tensor, Tensor) = nnmodel.forward_is(&[&[board], &[cond]]).try_into().unwrap();

    // Converting outputs into vectors
    let prob = prob.flatten(0, i64::try_from(prob.size().len()).unwrap() - 1);
    let prob = Vec::<f32>::try_from(prob).expect("Something went wrong when converting tensor into vector");
    let value = value.flatten(0, i64::try_from(value.size().len()).unwrap() - 1);
    let value = f32::try_from(value).expect("Could not convert value tensor to f32");




    let valid_actions_for_masking: Vec<f32> = generate_tile_plays(game_state).try_into().expect("could not convert mask into Vec<f32>"); 
    // This should output a correct valid moves depending on the variable game (-> whose turn it is)  


    let valid_actions = get_indices_of_ones(valid_actions_for_masking);

    if valid_actions.is_empty() {
        return (valid_actions, Vec::new(), 0.0)
    }

    let mut pi: Vec<f32> = prob.iter()
        .zip(valid_actions_for_masking.iter())
        .map(|(p, v)| p * v)
        .collect();

    let sum_probs = pi.sum();
    if sum_probs > 0 {
        pi /= sum_probs;                           // renormalize
    } else {                                                // Contingency for when all the actions with non-zero probabilities are masked
        let num_valid_actions = valid_actions.len();
        pi = valid_actions_for_masking / num_valid_actions;
    }
    return (valid_actions, pi, value)
}

fn get_improved_policy(root: Node) -> Vec<f32> {
    let pi_size = 7_usize.pow(4); //                NOTE: Assuming the board is 7x7 if not, change this
    let mut pi = vec![0.0; pi_size];
    let count_sum: f32 = root.children.values().map(|child| child.visits).sum();
    for (action, child) in &root.children {
        // let index = action_to_index(action); 
        let index: usize = action.try_into().unwrap();
        pi[index] = child.visits / count_sum;
    }
    pi
}

// This does a single mcts starting from whomever the current turn is assigned to  
pub fn mcts<T: BoardState>(nnmodel: CModule, game: Game<T>, iterations: usize) -> Vec<f32> {
    let (valid_actions, action_probs, reward) = model_predict(&game.state, &nnmodel);
    let root_board = board_to_matrix(&game.state.board);
    let num_valid_actions = valid_actions.len();
    let mut action_counts = HashMap::with_capacity(num_valid_actions);
    let mut action_Qs = HashMap::with_capacity(num_valid_actions);
    for action in valid_actions {
        action_counts.insert(action, 0.0);
        action_Qs.insert(action, 0.0);
    }

    let root = Node {
        board: root_board,
        parent: None,
        children: HashMap::with_capacity(num_valid_actions),
        visits: 1,
        valid_actions,
        action_probs,
        action_Qs,
        action_counts,
    };

    for _ in 0..iterations {
        let game_state_copy = game.state.Clone();
        let _ = search(game_state_copy, &mut root, &nnmodel);
    }

    get_improved_policy(root)
}
