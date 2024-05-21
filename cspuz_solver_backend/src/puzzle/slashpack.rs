use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::slashpack;

pub fn solve_slashpack(url: &str) -> Result<Board, &'static str> {
    let problem = slashpack::deserialize_problem(url).ok_or("invalid url")?;
    let ans = slashpack::solve_slashpack(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&ans));
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                if clue > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::Text("?")));
                }
            } else if let Some(a) = ans[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    match a {
                        slashpack::SLASHPACK_EMPTY => ItemKind::Dot,
                        slashpack::SLASHPACK_SLASH => ItemKind::Slash,
                        slashpack::SLASHPACK_BACKSLASH => ItemKind::Backslash,
                        _ => panic!(),
                    },
                ));
            }
        }
    }

    Ok(board)
}
