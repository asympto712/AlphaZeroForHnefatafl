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
// use pyo3::types::{PyFloat, PyList, PyString, PyTuple, PyInt};
// use numpy::array::{PyArray1, PyArray2};
use tch::CModule;

#[pyfunction]
fn self_play_function<'py> (nnmodel_path: &str, no_games: i32) 
-> PyResult<Vec<(Vec<Vec<u8>>, Vec<f32>, i32, i32)>> {

    let mut nnmodel = CModule::load(nnmodel_path).unwrap();
    nnmodel.set_eval();
    let data = self_play::self_play(nnmodel, no_games);

    // // Convert the vector of tuples into a Python list
    // let mut list = Vec::new();
    
    // for (board, pi, player, result) in data {
    //     let py_board = PyArray2::from_vec2(py, &board).unwrap();
    //     let py_pi = PyArray1::from_slice(py, &pi);

    //     // Create a Python tuple (matrix, float_vec, int1, int2)
    //     let tuple = (py_board, py_pi, player, result);
    //     let py_tuple = PyTuple::new(py, tuple).unwrap();
        
    //     list.push(py_tuple);
    // }
    // PyList::new(py, list)
    Ok(data)
}

#[pymodule]
#[pyo3(name="_azhnefatafl")]
fn _azhnefatafl(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(self_play_function, m)?)?;
    Ok(())
}