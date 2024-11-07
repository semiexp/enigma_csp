use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::statue_park::{self, StatueParkClue};

pub fn solve_statue_park(url: &str) -> Result<Board, &'static str> {
    let (problem, pieces) = statue_park::deserialize_problem(url).ok_or("invalid url")?;
    let ans = statue_park::solve_statue_park(&problem, &pieces).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&ans));
    for y in 0..height {
        for x in 0..width {
            match problem[y][x] {
                StatueParkClue::None => {
                    if let Some(a) = ans[y][x] {
                        board.push(Item::cell(
                            y,
                            x,
                            "green",
                            if a { ItemKind::Block } else { ItemKind::Dot },
                        ));
                    }
                }
                StatueParkClue::White => board.push(Item::cell(y, x, "black", ItemKind::Dot)),
                StatueParkClue::Black => board.push(Item::cell(y, x, "black", ItemKind::Fill)),
            }
        }
    }

    Ok(board)
}
