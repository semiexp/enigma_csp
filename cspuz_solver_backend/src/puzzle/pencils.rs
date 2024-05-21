use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::pencils::{self, PencilsAnswer, PencilsClue};

pub fn solve_pencils(url: &str) -> Result<Board, &'static str> {
    let problem = pencils::deserialize_problem(url).ok_or("invalid url")?;
    let (cell, line, border) = pencils::solve_pencils(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(
        BoardKind::OuterGrid,
        height,
        width,
        is_unique(&(&cell, &line)),
    );

    for y in 0..height {
        for x in 0..width {
            match &problem[y][x] {
                &PencilsClue::Num(n) => board.push(Item::cell(y, x, "black", ItemKind::Num(n))),
                &PencilsClue::Left => board.push(Item::cell(y, x, "black", ItemKind::PencilLeft)),
                &PencilsClue::Right => board.push(Item::cell(y, x, "black", ItemKind::PencilRight)),
                &PencilsClue::Up => board.push(Item::cell(y, x, "black", ItemKind::PencilUp)),
                &PencilsClue::Down => board.push(Item::cell(y, x, "black", ItemKind::PencilDown)),
                _ => {
                    if let Some(c) = cell[y][x] {
                        match c {
                            PencilsAnswer::Left => {
                                board.push(Item::cell(y, x, "green", ItemKind::PencilLeft))
                            }
                            PencilsAnswer::Right => {
                                board.push(Item::cell(y, x, "green", ItemKind::PencilRight))
                            }
                            PencilsAnswer::Up => {
                                board.push(Item::cell(y, x, "green", ItemKind::PencilUp))
                            }
                            PencilsAnswer::Down => {
                                board.push(Item::cell(y, x, "green", ItemKind::PencilDown))
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
                let kind = match (border.horizontal[y][x], line.vertical[y][x]) {
                    (Some(true), _) => Some(ItemKind::BoldWall),
                    (_, Some(true)) => Some(ItemKind::Line),
                    (Some(false), Some(false)) => Some(ItemKind::Cross),
                    _ => None,
                };
                if kind != Some(ItemKind::BoldWall) {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "#cccccc",
                        kind: ItemKind::Wall,
                    });
                }
                if let Some(kind) = kind {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind,
                    })
                }
            }
            if x < width - 1 {
                let kind = match (border.vertical[y][x], line.horizontal[y][x]) {
                    (Some(true), _) => Some(ItemKind::BoldWall),
                    (_, Some(true)) => Some(ItemKind::Line),
                    (Some(false), Some(false)) => Some(ItemKind::Cross),
                    _ => None,
                };
                if kind != Some(ItemKind::BoldWall) {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "#cccccc",
                        kind: ItemKind::Wall,
                    });
                }
                if let Some(kind) = kind {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "green",
                        kind,
                    })
                }
            }
        }
    }
    Ok(board)
}
