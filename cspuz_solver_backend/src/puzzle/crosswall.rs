use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::crosswall;

pub fn solve_crosswall(url: &str) -> Result<Board, &'static str> {
    let problem = crosswall::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = crosswall::solve_crosswall(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::DotGrid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some((size, level)) = problem[y][x] {
                if size > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(size)));
                }
                if level >= 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::NumLowerRight(level)));
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..=width {
            if let Some(b) = is_line.vertical[y][x] {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }
    for y in 0..=height {
        for x in 0..width {
            if let Some(b) = is_line.horizontal[y][x] {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }

    Ok(board)
}
