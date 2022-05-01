use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::nurimisaki;

pub fn solve_nurimisaki(url: &str) -> Result<Board, &'static str> {
    let problem = nurimisaki::deserialize_problem(url).ok_or("invalid url")?;
    let ans = nurimisaki::solve_nurimisaki(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Circle));
                if clue > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                }
            } else if let Some(a) = ans[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Dot } else { ItemKind::Fill },
                ));
            }
        }
    }

    Ok(board)
}
