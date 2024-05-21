use crate::board::{Board, BoardKind, FireflyDir, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::items::Arrow;
use cspuz_rs::puzzle::firefly;

pub fn solve_firefly(url: &str) -> Result<Board, &'static str> {
    let problem = firefly::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = firefly::solve_firefly(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Empty, height, width, is_unique(&is_line));

    for y in 0..(height - 1) {
        for x in 0..width {
            let mut need_default_edge = true;
            if let Some(b) = is_line.vertical[y][x] {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: if b {
                        ItemKind::BoldWall
                    } else {
                        ItemKind::Cross
                    },
                });
                need_default_edge = !b;
            }
            if need_default_edge {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "black",
                    kind: ItemKind::DottedWall,
                });
            }
        }
    }
    for y in 0..height {
        for x in 0..(width - 1) {
            let mut need_default_edge = true;
            if let Some(b) = is_line.horizontal[y][x] {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: if b {
                        ItemKind::BoldWall
                    } else {
                        ItemKind::Cross
                    },
                });
                need_default_edge = !b;
            }
            if need_default_edge {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: ItemKind::DottedWall,
                });
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if let Some((dir, n)) = problem[y][x] {
                let dir = match dir {
                    Arrow::Unspecified => panic!(),
                    Arrow::Up => FireflyDir::Up,
                    Arrow::Down => FireflyDir::Down,
                    Arrow::Left => FireflyDir::Left,
                    Arrow::Right => FireflyDir::Right,
                };
                board.push(Item {
                    y: y * 2,
                    x: x * 2,
                    color: "black",
                    kind: ItemKind::Firefly(dir, n),
                });
            }
        }
    }

    Ok(board)
}
