use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::litherslink;

pub fn solve_litherslink(url: &str) -> Result<Board, &'static str> {
    let problem = litherslink::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = litherslink::solve_litherslink(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::DotGrid, height, width, is_unique(&is_line));

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
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
