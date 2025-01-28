#![allow(non_snake_case)]

use std::collections::HashMap;
use rand::prelude::*;
use tch::{CModule, Tensor, Kind, Device};

type Action = u32;
type State = Vec<u32>;  // Here we assume the state is already converted into matrix representation & flattened 

const C_PUCT: f32 = 0.3;

#[derive(Debug, Clone, Copy)]
struct Node {
    state: State, 
    parent: Option<(Node<Action, State>, Action)>,
    children: HashMap<Action, Box<Node<Action, State>>>, 
    visits: f32,
    valid_actions: Vec<Action>, 
    action_probs: HashMap<Action, f32>,
    action_counts: HashMap<Action, f32>, 
    action_Qs: HashMap<Action, f32>,
}

impl Node{
    fn new(state: State,
           parent: Option<(Node<Action, State>, Action)>,
           valid_actions: Vec<Action>,
           visits: f32) -> Self {
        Node {
            state,
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
fn search(game: &mut Game, node: &mut Node, nnmodel: &CModule) -> f32 {

    if game.is_terminal(node.state) {
        let reward = game.player * game.reward(Node.state); // Assumes game.reward returns the reward from the ATTACKER's perspective
        return -1 * reward;
    }

    let action = node.valid_actions
            .iter()
            .max_by(|a, b| {
                node.uct_value(a).partial_cmp(&node.uct_value(b)).unwrap()
            })
            .unwrap();
    
    game.do_move(action);
        
    if !node.children.contains_key(action) {         
        let reward = expand(&mut node, &action, &game, &nnmodel);
        return -1 * reward
    }

    let next_node = **node.children.get(action).unwrap();
    let reward = search(game, &mut next_node, &nnmodel);

    node.actions_Qs(action) = (node.actions_counts(action) * node.actions_Qs(actions) + reward) / (node.actions_counts(action) + 1.0);
    node.actions_counts(action) += 1.0;
    node.visits += 1.0;
    return -1 * reward
    
}

fn expand(parent: &mut Node<Action, State>, action: &Action, game: &Game, nnmodel: &CModule) {
    let (valid_actions, action_probs, reward) = model_predict(game, nnmodel);
    let num_valid_actions = valid_actions.sum();
    let mut action_counts = HashMap::with_capacity(num_valid_actions);
    let mut action_Qs = HashMap::with_capacity(num_valid_actions);
    for action in valid_actions {
        action_counts.insert(*action, 0.0);
        action_Qs.insert(*action, 0.0);
    }

    let mut new_node: Node<Action, State> = Node{
        state: game.state,
        parent: (parent, action),
        children: HashMap::new(),
        visits: 1.0,
        valid_actions,
        action_probs,
        action_counts,
        action_Qs,
    };
    parent.children.insert(action, Box::new(new_node));

    return reward
}

fn model_predict(game: &Game, nnmodel: &CModule) -> (Vec<f32>, Vec<f32>, f32) {

    // Preparing input Tensors
    let board = Tensor::of_data(game.state, &[7,7], (Kind::Float32, Device::Cpu));
    let cond: [bool; 1] = if game.player == 1 {[true]} else {[false]};
    let cond: Tensor = Tensor::from_slice(&cond);

    let (prob, reward): (Tensor, Tensor) = nnmodel.forward_is(&[&[board], &[cond]]).try_into().unwrap();

    // Converting outputs into vectors
    let prob = prob.flatten(0, i64::try_from(prob.size().len()).unwrap() - 1);
    let prob = Vec::<f32>::try_from(prob).expect("Something went wrong when converting tensor into vector");
    let reward = reward.flatten(0, i64::try_from(reward.size().len()).unwrap() - 1);
    let reward = f32::try_from(reward).expect("Could not convert value tensor to f32");

    let valid_actions = game.get_valid_actions(game.state);
    let valid_actions_for_masking: Vec<f32> = game.get_valid_actions_for_masking(game.state).try_into().expect("could not convert mask into Vec<f32>"); 
    // This should output a correct valid moves depending on the variable game (-> whose turn it is)  

    let mut action_probs: Vec<f32> = prob.iter()
        .zip(valid_actions_for_masking.iter())
        .map(|(p, v)| p * v)
        .collect();

    let sum_probs = actions_probs.sum();
    if sum_probs > 0 {
        actions_probs /= sum_probs;                           // renormalize
    } else {                                                // Contingency for when all the actions with non-zero probabilities are masked
        let num_valid_actions = valid_actions_for_masking.sum();
        actions_probs = valid_actions_for_masking / num_valid_actions;
    }
    return (valid_actions, action_probs, reward)
}

fn get_improved_policy(root: Node) -> Vec<f32> {
    let pi_size = powd(game.size, 4);
    let mut pi = vec![0.0; pi_size];
    let count_sum: f32 = root.children.values().map(|child| child.visits).sum();
    for (action, child) in &root.children {
        let index = action_to_index(action); 
        pi[index] = child.visits / count_sum;
    }
    pi
}

// This does a single mcts starting from whomever the current turn is assigned to  
fn mcts(nnmodel: CModule, game: Game, iterations: usize) -> Vec<f32> {
    let (valid_actions, action_probs, reward) = model_predict(game, &nnmodel);
    let root_state = game.state.copy();
    let num_valid_actions = valid_actions.sum();
    let mut action_counts = HashMap::with_capacity(num_valid_actions);
    let mut actions_Qs = HashMap::with_capacity(num_valid_actions);
    for action in valid_actions {
        action_counts.insert(action, 0.0);
        action_Qs.insert(action, 0.0);
    }

    let root = Node {
        state: root_state,
        parent: None,
        children: HashMap::with_capacity(num_valid_actions),
        visits: 1,
        valid_actions,
        action_probs,
        action_Qs,
        action_counts,
    };

    for _ in 0..iterations {
        let game_copy = game.copy();
        let _ = search(game_copy, &mut root, &nnmodel);
    }

    get_improved_policy(root)
}
