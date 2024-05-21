use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::coral;

pub fn solve_coral(url: &str) -> Result<Board, &'static str> {
    let (vertical, horizontal) = coral::deserialize_problem(url).ok_or("invalid url")?;
    let is_black = coral::solve_coral(&vertical, &horizontal).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_black));

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
