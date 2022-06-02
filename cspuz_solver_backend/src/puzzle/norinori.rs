use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::norinori;

pub fn solve_norinori(url: &str) -> Result<Board, &'static str> {
    let borders = norinori::deserialize_problem(url).ok_or("invalid url")?;
    let is_black = norinori::solve_norinori(&borders).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(board)
}
