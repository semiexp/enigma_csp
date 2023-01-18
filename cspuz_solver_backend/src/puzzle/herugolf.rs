use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::herugolf;

pub fn solve_herugolf(url: &str) -> Result<Board, &'static str> {
    let (pond, clues) = herugolf::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = herugolf::solve_herugolf(&pond, &clues).ok_or("no answer")?;

    let height = pond.len();
    let width = pond[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    for y in 0..height {
        for x in 0..width {
            if pond[y][x] {
                board.push(Item::cell(y, x, "#ccccff", ItemKind::Fill));
            }
            if let Some(n) = clues[y][x] {
                if n > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Circle));
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::Text("H")));
                }
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", None);

    Ok(board)
}
