use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::shakashaka::{self, ShakashakaCell};

pub fn solve_shakashaka(url: &str) -> Result<Board, &'static str> {
    let problem = shakashaka::deserialize_problem(url).ok_or("invalid url")?;
    let answer = shakashaka::solve_shakashaka(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Fill));
                if n >= 0 {
                    board.push(Item::cell(y, x, "white", ItemKind::Num(n)));
                }
            } else if let Some(a) = answer[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    match a {
                        ShakashakaCell::Blank => ItemKind::Dot,
                        ShakashakaCell::UpperLeft => ItemKind::AboloUpperLeft,
                        ShakashakaCell::UpperRight => ItemKind::AboloUpperRight,
                        ShakashakaCell::LowerLeft => ItemKind::AboloLowerLeft,
                        ShakashakaCell::LowerRight => ItemKind::AboloLowerRight,
                    },
                ));
            }
        }
    }

    Ok(board)
}
