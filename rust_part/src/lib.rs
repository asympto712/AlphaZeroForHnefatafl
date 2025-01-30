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
use pyo3::types::{PyFloat, PyList, PyString, PyTuple};
use numpy::{IntoPyArray, PyArray2};
use tch::CModule;

#[pyfunction]
fn self_play_function(nnmodel_path: PyString, no_games: PyInt, py: Py<'_>) 
-> PyResult<PyList<PyTuple<PyArray2<PyInt>, PyList<PyFloat>, PyInt, PyInt>>, PyErr> {

    let path: &str = nnmodel_path.extract()?;
    let nnmodel = CModule::load(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load model: {}", e)))?;
    let data = self_play::self_play(nnmodel, 100);

    // Convert the vector of tuples into a Python list
    let py_list = PyList::empty(py);
    
    for (board, pi, player, result) in data {
        let py_board = PyArray2::from_vec2(py, &board)?;
        let py_pi = PyArray::from_slice(py, &pi)?;

        // Create a Python tuple (matrix, float_vec, int1, int2)
        let py_tuple = PyTuple::new(py, &[np_board, np_pi, player.into_py(py), result.into_py(py)]);
        
        py_list.append(py_tuple).unwrap();
    }
    Ok(py_list.into(PyObject))
}

#[pymodule]
fn alpha_zero_for_hnefatafl(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(self_play_function, m)?)?;
    Ok(())
}