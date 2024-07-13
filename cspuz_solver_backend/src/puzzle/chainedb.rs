use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::chainedb;

pub fn solve_chainedb(url: &str) -> Result<Board, &'static str> {
    let problem = chainedb::deserialize_problem(url).ok_or("invalid url")?;
    let ans = chainedb::solve_chainedb(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&ans));
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Fill));
                if clue > 0 {
                    board.push(Item::cell(y, x, "white", ItemKind::Num(clue)));
                } else {
                    board.push(Item::cell(y, x, "white", ItemKind::Text("?")));
                }
            } else if let Some(a) = ans[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(board)
}
