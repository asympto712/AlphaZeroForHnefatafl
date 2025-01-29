#![allow(unused_imports)]

use crate::hnefgame::tiles::{Axis, Tile};
use crate::hnefgame::game::Game;
use crate::hnefgame::play::Play;
use crate::hnefgame::pieces::{PieceType, Side};
use crate::hnefgame::game::state::GameState;
use crate::hnefgame::board::state::BoardState;

use std::fs::{OpenOptions, read_to_string};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::io::stdin;
use std::str::FromStr;




// takes game as input and returns a string vector of all legal moves
pub fn get_all_possible_moves<T: BoardState>(game: &Game<T>) -> Vec<String> {
    let mut possible_moves = Vec::new();
    for tile in game.state.board.iter_occupied(game.state.side_to_play) {
        if let Ok(mut iter) = game.iter_plays(tile) {
            while let Some(valid_play) = iter.next() {
                possible_moves.push(convert_play_to_notation(&valid_play.play));
            }
        }
    }
    possible_moves
}


// takes game and a vector of moves as input and returns a binary vector 
pub fn validate_moves<T: BoardState>(game: &Game<T>, possible_moves: Vec<Play>) -> Vec<u8> {
    possible_moves.iter().map(|play| {
        match game.logic.validate_play(*play, &game.state) {
            Ok(_) => 1,
            Err(_) => 0,
        }
    }).collect()
}

//convers a move from a Play to a string notation
pub fn convert_play_to_notation(play: &Play) -> String {
    let start_tile = play.from;
    let (start_row, start_col) = (start_tile.row, start_tile.col);

    let (end_row, end_col) = match play.movement.axis {
        Axis::Horizontal => (start_row, (start_col as i8 + play.movement.displacement) as u8),
        Axis::Vertical => ((start_row as i8 + play.movement.displacement) as u8, start_col),
    };

    let start_col_char = (start_col as u8 + b'a') as char;
    let start_row_num = (start_row + 1).to_string();

    let end_col_char = (end_col as u8 + b'a') as char;
    let end_row_num = (end_row + 1).to_string();

    format!("{}{}-{}{}", start_col_char, start_row_num, end_col_char, end_row_num)
}


// converts a board state to a matrix with piece values
pub fn board_to_matrix<T: BoardState>(game_state: &GameState<T>) -> Vec<Vec<u8>> {
    let side_len = game_state.board.side_len();
    let mut matrix = vec![vec![0; side_len as usize]; side_len as usize];

    // Initialize special tiles
    matrix[0][0] = 20;
    matrix[0][(side_len - 1) as usize] = 20;
    matrix[(side_len - 1) as usize][0] = 20;
    matrix[(side_len - 1) as usize][(side_len - 1) as usize] = 20;
    matrix[(side_len / 2) as usize][(side_len / 2) as usize] = 30;

    // Iterate over the board and add piece values
    for row in 0..side_len {
        for col in 0..side_len {
            let tile = Tile::new(row, col);
            if let Some(piece) = game_state.board.get_piece(tile) {
                let value = match piece.piece_type {
                    PieceType::Soldier => 1,
                    PieceType::Knight => 2,
                    PieceType::King => 5,
                    _ => 0,
                };
                matrix[row as usize][col as usize] += value;
            }
        }
    }

    matrix
}


// takes in path, matrix of a board, policy vector, current player and game end value
// checks the length of the file and deletes as neccesary
pub fn write_to_file(
    file_path: &str,
    matrix: Vec<Vec<u8>>,
    vector: Vec<u8>,
    value1: u8,
    value2: u8,
    max_entries: usize,
) -> std::io::Result<()> {
    let path = Path::new(file_path);
    let mut entries = Vec::new();

    // Read existing entries if the file exists
    if path.exists() {
        let content = read_to_string(file_path)?;
        entries = content.lines().map(|line| line.to_string()).collect();
    }

    // Check if the number of entries exceeds max_entries
    if entries.len() >= max_entries {
        entries.remove(0); // Remove the oldest entry
    }

    // Create a new entry
    let new_entry = format!(
        "{}\n{}\n{}\n{}",
        matrix
            .iter()
            .map(|row| row.iter().map(|&v| v.to_string()).collect::<Vec<_>>().join(","))
            .collect::<Vec<_>>()
            .join("\n"),
        vector.iter().map(|&v| v.to_string()).collect::<Vec<_>>().join(","),
        value1,
        value2
    );

    // Add the new entry to the entries list
    entries.push(new_entry);

    // Write all entries back to the file
    let file = OpenOptions::new().write(true).create(true).truncate(true).open(file_path)?;
    let mut writer = BufWriter::new(file);
    for entry in entries {
        writeln!(writer, "{}", entry)?;
    }

    Ok(())
}


pub fn generate_tile_plays<T: BoardState>(game: &Game<T>) -> Vec<i8> {
    let mut tiles = Vec::new();
    let columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
    let rows = [1, 2, 3, 4, 5, 6, 7];

    // Initialize tiles vector with strings of the form letter_number
    for &col in &columns {
        for &row in &rows {
            tiles.push(format!("{}{}", col, row));
        }
    }
    let mut result = Vec::new();

    // Iterate over each element of the tiles vector
    for start_tile in &tiles {
        let mut binary_vector = Vec::new();

        for end_tile in &tiles {
            // Convert the strings to a play
            let play_str = format!("{}-{}", start_tile, end_tile);
            let play = match Play::from_str(&play_str) {
                Ok(play) => play,
                Err(_e) => {
                    binary_vector.push(0);
                    continue;
                }
            };

            let valid = match game.state.side_to_play {
                Side::Attacker => game.logic.validate_play_for_side(play, Side::Attacker, &game.state),
                Side::Defender => game.logic.validate_play_for_side(play, Side::Defender, &game.state),
            };

            if valid.is_ok() {
                binary_vector.push(1);
            } else {
                binary_vector.push(0);
            }
        }
        result.push(binary_vector);
    }

    let mut single_vector  = Vec::new();

    for row in &result {
        for element in row {
            single_vector.push(*element);
        }
    }
    return single_vector
}


pub fn get_indices_of_ones(binary_vector: &Vec<i8>) -> Vec<usize> {
    let mut indices = Vec::new();
    for (index, &value) in binary_vector.iter().enumerate() {
        if value == 1 {
            indices.push(index);
        }
    }
    indices
}


pub fn action_to_str(action : &u32) -> String {
    let r = action % 49;
    let q = (action / 49) as f32;
    let start: String = int_to_str(q.floor() as u32);
    let end: String = int_to_str(r);
    format!("{}-{}", start, end)
}

fn int_to_str(u: u32) -> String {
    let table = ["a","b","c","d","e","f","g"];
    let r = u % 7;
    let q = (u / 7) as f32;
    let row: String = (q.floor() as u32 + 1).to_string();
    let col: &str = table[r as usize];
    row + col
}

pub fn input(prompt: &str) -> std::io::Result<String> {
    println!("{prompt}");
    let mut s: String = String::new();
    stdin().read_line(&mut s)?;
    Ok(s.trim().to_string())
}

pub fn get_play() -> Option<Play> {
    loop {
        if let Ok(m_str) = input("Please enter your move (or type 'exit' to quit):") {
            if m_str.to_lowercase() == "exit" {
                return None;
            }
            match Play::from_str(&m_str) {
                Ok(play) => return Some(play),
                Err(e) => println!("Invalid move ({e:?}). Try again.")
            }
        } else {
            println!("Error reading input. Try again.");
        }
    }
}
