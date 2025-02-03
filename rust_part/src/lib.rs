pub mod self_play;
pub mod support;
pub mod mcts;
pub mod hnefgame {
    pub mod board;
    pub mod game;
    pub mod bitfield;
    pub mod error;
    pub mod pieces;
    pub mod play;
    pub mod preset;
    pub mod rules;
    pub mod tiles;
    pub mod utils;
}

use pyo3::prelude::*;
use tch::CModule;

#[pyfunction]
fn self_play_function<'py> (nnmodel_path: &str, no_games: i32, mcts_iterations: u32, verbose: bool) 
-> PyResult<Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)>> {

    let mut nnmodel = 
    if tch::Cuda::is_available() {
        CModule::load_on_device(nnmodel_path, tch::Device::Cuda(0)).unwrap()
    } else {
        CModule::load_on_device(nnmodel_path, tch::Device::Cpu).unwrap()
    };
    
    nnmodel.set_eval();
    let data = self_play::self_play(nnmodel, no_games, mcts_iterations, verbose);
    Ok(data)
}

#[pymodule]
#[pyo3(name="_azhnefatafl")]
fn _azhnefatafl(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(self_play_function, m)?)?;
    Ok(())
}