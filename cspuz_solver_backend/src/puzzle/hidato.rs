use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::hidato;

pub fn solve_hidato(url: &str) -> Result<Board, &'static str> {
    let problem = hidato::deserialize_problem(url).ok_or("invalid url")?;
    let answer = hidato::solve_hidato(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&answer));

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                if n == -1 {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                }
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
