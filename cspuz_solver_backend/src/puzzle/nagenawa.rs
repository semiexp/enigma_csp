use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::graph;
use cspuz_rs::puzzle::nagenawa;

pub fn solve_nagenawa(url: &str) -> Result<Board, &'static str> {
    let (borders, clues) = nagenawa::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = nagenawa::solve_nagenawa(&borders, &clues).ok_or("no answer")?;

    let height = is_line.horizontal.len();
    let width = is_line.horizontal[0].len() + 1;
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    board.add_borders(&borders, "black");
    board.add_lines_irrefutable_facts(&is_line, "green", None);

    let rooms = graph::borders_to_rooms(&borders);
    assert_eq!(rooms.len(), clues.len());
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let (y, x) = rooms[i][0];
            board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
        }
    }

    Ok(board)
}
