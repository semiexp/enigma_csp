use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::nurimaze;

pub fn solve_nurimaze(url: &str) -> Result<Board, &'static str> {
    let (borders, clues) = nurimaze::deserialize_problem(url).ok_or("invalid url")?;
    let is_black = nurimaze::solve_nurimaze(&borders, &clues).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_black));

    for y in 0..height {
        for x in 0..width {
            match clues[y][x] {
                1 => board.push(Item::cell(y, x, "black", ItemKind::Text("S"))),
                2 => board.push(Item::cell(y, x, "black", ItemKind::Text("G"))),
                3 => board.push(Item::cell(y, x, "black", ItemKind::Circle)),
                4 => board.push(Item::cell(y, x, "black", ItemKind::Triangle)),
                _ => (),
            }

            if let Some(b) = is_black[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Fill } else { ItemKind::Dot },
                ));
            }
        }
    }

    board.add_borders(&borders, "black");

    Ok(board)
}
