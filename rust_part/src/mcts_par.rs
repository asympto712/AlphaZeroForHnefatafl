use crate::hnefgame::game::Game;
use crate::hnefgame::game::GameOutcome::{Draw, Win};
use crate::hnefgame::game::GameStatus::{Ongoing, Over};
use crate::hnefgame::game::state::GameState;
use crate::hnefgame::play::Play;
use crate::hnefgame::pieces::Side;
use crate::hnefgame::board::state::BoardState;
use crate::hnefgame::game::logic::GameLogic;
use super::support::{action_to_str, board_to_matrix, generate_tile_plays, get_ai_play, get_indices_of_ones};

use std::collections::HashMap;
use rv::traits::Sampleable;
use tch::{CModule, Tensor, Device, IValue};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;
use threadpool::ThreadPool;
use std::sync::{Arc,Mutex};
use rv::prelude::Dirichlet;
use rand::thread_rng;



type Action = u32;

const C_PUCT: f32 = 0.3;

pub struct Tree<T: BoardState + Send>{
    pub refs: Vec<Rc<RefCell<Node>>>,
    game_logic: GameLogic,
    root_game_state: GameState<T>,
}
pub struct Notr {
    num: usize,
    parent: Option<(Action, usize)>,
    children: HashMap<Action, usize>, 
    visits: f32,
    valid_actions: Vec<Action>,
    action_counts: HashMap<Action, f32>, 
    action_probs: HashMap<Action, f32>,
    action_qs: HashMap<Action, f32>,
    pub depth: u32,
}

#[allow(dead_code)]
pub struct Term {
    num: usize,
    parent: Option<(Action, usize)>,
    value: f32,
    pub depth: u32,
}

#[allow(dead_code)]
pub enum Node {
    Notr(Notr),
    Term(Term),
}


impl Notr{
    fn new(
        num: usize,
        parent: usize,
        action: &Action,
        visits: f32,
        valid_actions: Vec<Action>,
        pi: Vec<f32>,
        depth: u32)
        -> Self {

        let mut action_counts = HashMap::with_capacity(valid_actions.len());
        let mut action_probs = HashMap::with_capacity(valid_actions.len());
        let mut action_qs = HashMap::with_capacity(valid_actions.len());

        for action in valid_actions.iter() {
            action_counts.insert(*action, 0.0);
            action_probs.insert(*action, pi[*action as usize]);
            action_qs.insert(*action, 0.0);
        }

         Self{
            num,
            parent: Some((*action, parent)),
            children: HashMap::new(),
            visits,
            valid_actions,
            action_counts,
            action_probs,
            action_qs,
            depth,
        }
    }

    fn add_child(&mut self, action: Action, child: usize) {
        self.children.insert(action, child);
    }

    fn uct_value(&self, action: &Action) -> f32 {
        
        let q = self.action_qs.get(action).unwrap();
        let count = self.action_counts.get(action).unwrap();
        let p = self.action_probs.get(action).unwrap();
        return *q + C_PUCT * p * self.visits.sqrt() / (1.0 + *count);
    }

    pub fn display_info(&self) {
        println!("Number of valid actions: {}", self.valid_actions.len());
        let mut action_counts: Vec<_> = self.action_counts.iter().collect();
        action_counts.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        println!("Top 20 actions with the highest counts:");
        for (action, count) in action_counts.iter().take(20) {
            println!("Action: {}, Count: {}", action_to_str(action), count);
        }
        println!("Depth is: {}", self.depth);
        match &self.parent {
            Some((action, parent)) => println!("Parent Action: {}, Parent Node: {}", action_to_str(action), parent),
            None => println!("This node has no parent."),
        }
        println!("Children are...");
        for (action, child) in &self.children {
            println!("Action: {}, Child: {}", action_to_str(action), child);
        }
    }
}

impl Term {
    fn new(num: usize, parent: usize, action: &Action, value: f32, depth: u32) -> Self {
        Self { num, parent: Some((*action, parent)), value, depth}
    }
}

impl<T: BoardState + Send + 'static> Tree<T> {
    pub fn new(game: &Game<T>, nnmodel: Arc<CModule>) -> Self {

        let game_logic = game.logic.clone();
        let root_game_state = game.state.clone();
        let (valid_actions, pi, _) = model_predict(&game.state, &nnmodel, &game_logic);
        let mut action_counts = HashMap::with_capacity(valid_actions.len());
        let mut action_probs = HashMap::with_capacity(valid_actions.len());
        let mut action_qs = HashMap::with_capacity(valid_actions.len());

        for action in valid_actions.iter() {
            action_counts.insert(*action, 0.0);
            action_probs.insert(*action, pi[*action as usize]);
            action_qs.insert(*action, 0.0);
        }

        let root = Notr{
            num: 0,
            parent: None,
            children: HashMap::with_capacity(valid_actions.len()),
            visits: 0.0,
            valid_actions,
            action_counts,
            action_probs,
            action_qs,
            depth: 0
        };

        let mut refs: Vec<Rc<RefCell<Node>>> = Vec::new();
        refs.push(Rc::new(RefCell::new(Node::Notr(root))));
        Self { refs, game_logic, root_game_state}
    }

    fn add_notr(&mut self,
        parent_num: usize,
        action: Action,
        valid_actions: Vec<Action>,
        pi: Vec<f32>) 
        -> usize {

        let mut par_node = self.refs[parent_num].borrow_mut();
        let parent = match &mut *par_node {
            Node::Notr(notr) => notr,
            _ => panic!("Parent must be a Node::Notr"),
        };
        let par_depth = parent.depth;
        let num = self.refs.len();
    
        let new_notr = Notr::new(num, parent_num, &action, 0.0, valid_actions, pi, par_depth + 1);
        parent.add_child(action, num);
        drop(par_node);
        self.refs.push(Rc::new(RefCell::new(Node::Notr(new_notr))));
        num
    }

    fn add_term(&mut self, parent_num: usize, action: Action) -> usize{
        let mut par_node = self.refs[parent_num].borrow_mut();
        let parent = match &mut *par_node {
            Node::Notr(notr) => notr,
            _ => panic!("Parent must be a Node::Notr"),
        };
        let value: f32 = 1.0;
        let par_depth = parent.depth;
        let num = self.refs.len();
        let new_term = Term::new(num, parent_num, &action, value, par_depth + 1);
        parent.add_child(action, num);
        drop(par_node);
        self.refs.push(Rc::new(RefCell::new(Node::Term(new_term))));
        num
    }

    // Add Dirichlet noise to the actions from the root node.
    fn root_dirichlet(&self) {
        let mut root = self.refs[0].borrow_mut();
        if let Node::Notr(root) = &mut *root{
            let num_valid_actions = root.valid_actions.len();
            let dir = Dirichlet::symmetric(0.3, num_valid_actions).unwrap();
            let mut rng = thread_rng();
            let dir_values: Vec<f64> = dir.draw(&mut rng);
            for (i, action) in root.valid_actions.iter().enumerate() {
                let noise = dir_values[i] as f32;
                let p = root.action_probs.get_mut(action).unwrap();
                *p = (1.0 - 0.25) * *p + 0.25 * noise;
            }
        }

    }
    

    //This function starts from a node, traverse the tree until it finds an unseen (pre)node,
    //and returns the tuple of (leaf node, action) just before the unseen node.
    fn explore(&self, node: usize, game_state: GameState<T>) -> Option<(usize, Action, GameState<T>)> {
        match &*self.refs[node].borrow() {
            Node::Term(_) => None,
            Node::Notr(notr) => {
                // let mut rng = rng();
                // let action = notr.valid_actions.choose(&mut rng).unwrap();
                let cur_valid_actions = notr.valid_actions.clone();
                let action = cur_valid_actions
                .iter()
                .max_by(|a,b| {
                    notr.uct_value(a).partial_cmp(&notr.uct_value(b)).unwrap()
                })
                .unwrap();

                let play_string = action_to_str(action);
                let play: Play = get_ai_play(&play_string);
                let _ = self.game_logic.do_play(play, game_state);

                let next_node = notr.children.get(action);
                let num = notr.num;
                match next_node {
                    None => Some((num, *action, game_state)),
                    Some(node) => self.explore(*node, game_state),
                }
            }
        }
    }

    // Starting from the new leaf, update the node's parent's visits and action counts.
    fn backup(&self, notr_num: usize, reward: f32){
        let node = self.refs[notr_num].borrow_mut();
        if let Node::Notr(notr) = &*node{
            match notr.parent{
                None => return,
                Some((par_action, par_n)) => {
                    let mut parent = self.refs[par_n].borrow_mut();
                    if let Node::Notr(par) = &mut *parent {
                        par.visits += 1.0;
                        // par.action_counts.entry(par_action).and_modify(|e| {*e += 1.0});
                        if let Some(q) = par.action_qs.get_mut(&par_action) {
                            if let Some(count) = par.action_counts.get(&par_action) {
                                *q = (*count * *q + reward) / (*count + 1.0);
                            }
                        }
                    
                        if let Some(result) = par.action_counts.get_mut(&par_action) {
                            *result += 1.0;
                        }
                    }
                    drop(parent);
                    self.backup(par_n, -1.0 * reward);
                }
            }
        } 
    }


    pub fn mcts_par(&mut self, nnmodel: Arc<CModule>, num_iter: usize) -> Vec<f32> {
        let pool = ThreadPool::new(2);
        let (tx, rx) = mpsc::channel();
        let tx = Arc::new(Mutex::new(tx));
        // let shared_nnmodel = Arc::new(nnmodel);
        let shared_logic = Arc::new(self.game_logic);

        for _ in 0..num_iter{
            
            let mut leaves: Vec<(usize, Action, GameState<T>)> = Vec::with_capacity(2);
            let mut count: u8 = 0;
            // We want to use separate threads to run inference using the NN model,
            // but it is possible that we don't encounter leaf node that is not a terminal node for a long time.
            // The additional condition [count < 10] is to avoid such cases. 
            while leaves.len() < leaves.capacity() && count < 10{
                let root_state = self.root_game_state.clone();
                self.root_dirichlet();
                let result = self.explore(0, root_state);
                match result{
                    None => (),
                    Some((notr_num, action, game_state)) => {
                        //Replace this with the game logic to determine if the new node is a terminal
                        // let notr_node = self.refs[notr_num].borrow();
                        
                        let option = calc_reward(game_state, self.game_logic);
                        match option {
                            Some(reward) => {
                                let leaf_num = self.add_term(notr_num, action);
                                self.backup(leaf_num, reward);
                            },
                            None => {
                                leaves.push((notr_num, action, game_state));
                            }
                        }

                        // if let Node::Notr(notr) = &*notr_node {
                        //     if notr.depth == DEPTH_LIMIT{
                        //         drop(notr_node);
                        //         let leaf_num = self.add_term(notr_num, action);
                        //         let reward = 1.0;
                        //         self.backup(leaf_num, reward);
                        //     } else {
                        //         leaves.push((notr_num, action, game_state));
                        //     }
                        // } else {
                        //     panic!("Tree.explore should have returned Term, which is not supposed to happen!")
                        // }
                    }
                }
                count += 1;
                // println!("count: {}", count);
            }
            
            let leaves_length = leaves.len();

            for (old_leaf, action, game_state) in leaves{

                // println!("{}, {}", old_leaf, action);

                // set up threads to run inference, send the outputs
                let tx = Arc::clone(&tx);
                let game_logic = Arc::clone(&shared_logic);
                // let nnmodel = Arc::clone(&shared_nnmodel);
                let nnmodel = Arc::clone(&nnmodel);

                pool.execute(move || {
                    // println!("Hello from the spawned thread!");
                    let (valid_actions, pi, value) = model_predict(&game_state, &nnmodel, &game_logic);
                    tx.lock().unwrap().send((old_leaf, action, valid_actions, pi, value)).expect("Failed to send message");
                });
            }

            for (old_leaf, action, valid_actions, pi, value) in rx.iter().take(leaves_length) {
                // println!("{}, {}, {}", old_leaf, action, reward);
                let new_leaf = self.add_notr(old_leaf, action, valid_actions, pi);
                self.backup(new_leaf, value);
                // println!("backup complete");
            }
            // println!("...");
        }

        // let root = &*self.refs[0].borrow();
        // if let Node::Notr(notr) = root {
        //     Ok(notr.action_counts.clone())
        // } else {
        //     Err("The root of the tree should be a notr".to_string())
        // }

        self.get_improved_policy()
    }

    fn get_improved_policy(&self) -> Vec<f32>{
        let root = self.refs[0].borrow();
        let policy_size = 7_usize.pow(4);
        let mut policy = vec![0.0; policy_size];
        if let Node::Notr(notr) = &*root{
            let count_sum: f32 = notr.action_counts.values().sum();
            for (action, count) in &notr.action_counts{
                policy[*action as usize] = count / count_sum;
            }
        }
        policy
    }
}


fn calc_reward<T:BoardState>(game_state: GameState<T>, game_logic: GameLogic) -> Option<f32> {
    let current_player = game_state.side_to_play;
    match game_state.status {
        Ongoing => {
            let num_valid_actions: i32 = generate_tile_plays(&game_logic, &game_state).iter().map(|&x| x as i32).sum();
            if num_valid_actions <= 0 {
                return Some(0.0)
            } else { return None}
        },
        Over(outcome) => match outcome {
            Win(_, side) => {
                if current_player ==  side {
                    return Some(1.0)
                } else {
                    return Some(-1.0)
                }
            }
            Draw(_) => return Some(0.0)
        },
    }
}

fn model_predict<T: BoardState>(game_state: &GameState<T>, nnmodel: &CModule, game_logic: &GameLogic)
-> (Vec<u32>, Vec<f32>, f32) {

    // Preparing input Tensors
    // let matrix_representation = board_to_matrix(game_state);
    let matrix_representation: Vec<Vec<f32>> = board_to_matrix(game_state)
    .iter()
    .map(|row| row.iter().map(|&x| x as f32).collect())
    .collect();

    let board = Tensor::from_slice2(&matrix_representation); //Assuming game.state is a 2-dimensional vector
    let cond: [bool; 1] = match game_state.side_to_play {
        Side::Attacker => [true],
        Side::Defender => [false],
    };

    let cond: Tensor = Tensor::from_slice(&cond);

    // Run the inference using the nnmodel
    // let input = IValue::Tuple(vec![IValue::Tensor(board), IValue::Tensor(cond)]);
    // let output = nnmodel.forward_is(&[input]);
    
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    let board = board.to_device(device);
    let cond = cond.to_device(device);
    let output = nnmodel.forward_is(&[IValue::Tensor(board), IValue::Tensor(cond)]);
    let (log_prob, value) = match output {
        Ok(IValue::Tuple(output)) => {
            if output.len() != 2 {
                panic!("Expected tuple of 2 tensors, but got {}", output.len());
            }

            let out1 = match &output[0] {
                IValue::Tensor(t) => t.shallow_clone(),
                _ => panic!("Expected a tensor as the first output"),
            };

            let out2 = match &output[1] {
                IValue::Tensor(t) => t.shallow_clone(),
                _ => panic!("Expected a tensor as the second output"),
            };

            (out1, out2)
        },
        _ => panic!("unexpected output from the model"),
    };

    // Converting outputs into vectors
    let log_prob = log_prob.flatten(0, i64::try_from(log_prob.size().len()).unwrap() - 1);
    let log_prob = Vec::<f32>::try_from(log_prob).expect("Something went wrong when converting tensor into vector");
    let value = value.flatten(0, i64::try_from(value.size().len()).unwrap() - 1);
    let value = f32::try_from(value).expect("Could not convert value tensor to f32");

    // NOTE: The NN model is trained to always predict the value from the Attacker's perspective. 
    // Therefore, in MCTS, we need to adjust the value to the current player's perspective.

    let value = match game_state.side_to_play {
        Side::Attacker => value,
        Side::Defender => -1.0 * value,
    };

    let valid_actions_for_masking: Vec<i8> = generate_tile_plays(game_logic, game_state); 
    // This should output a correct valid moves depending on the variable game (-> whose turn it is)  

    let valid_actions = get_indices_of_ones(&valid_actions_for_masking);
    let valid_actions: Vec<u32> = valid_actions.iter().map(|&x| x.try_into().expect("could not convert action into u32")).collect();

    if valid_actions.is_empty() {
        return (valid_actions, Vec::new(), 0.0)
    }

    let mut pi: Vec<f32> = log_prob.iter()
        .zip(valid_actions_for_masking.iter())
        .map(|(p, v)| p.exp() * (*v as f32))  // nnmodel outputs log_softmax, so we need to apply .exp() to recover the original probability
        .collect();

    let sum_probs: f32 = pi.iter().sum();

    if sum_probs > 0.0 {
        for p in &mut pi {
            *p /= sum_probs;
        }                           // renormalize
    } else {                                                // Contingency for when all the actions with non-zero probabilities are masked
        let num_valid_actions = valid_actions.len();
        pi = valid_actions_for_masking.iter().map(|&v| v as f32 / num_valid_actions as f32).collect();
    }

    return (valid_actions, pi, value)
}
