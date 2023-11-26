use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::tontonbeya;

fn num_to_item(n: i32) -> ItemKind {
    match n {
        0 => ItemKind::Circle,
        1 => ItemKind::Triangle,
        2 => ItemKind::Square,
        _ => panic!(),
    }
}

pub fn solve_tontonbeya(url: &str) -> Result<Board, &'static str> {
    let (borders, clues) = tontonbeya::deserialize_problem(url).ok_or("invalid url")?;
    let answer = tontonbeya::solve_tontonbeya(&borders, &clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);
    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = clues[y][x] {
                board.push(Item::cell(y, x, "black", num_to_item(n)));
            } else if let Some(n) = answer[y][x] {
                board.push(Item::cell(y, x, "green", num_to_item(n)));
            }
        }
    }

    Ok(board)
}
