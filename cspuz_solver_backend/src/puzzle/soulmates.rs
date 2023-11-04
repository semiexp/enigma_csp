use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::soulmates;

pub fn solve_soulmates(url: &str) -> Result<Board, &'static str> {
    let problem = soulmates::deserialize_problem(url).ok_or("invalid url")?;
    let answer = soulmates::solve_soulmates(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            } else if let Some(n) = answer[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if n == 0 {
                        ItemKind::Dot
                    } else {
                        ItemKind::Num(n)
                    },
                ));
            }
        }
    }

    Ok(board)
}
