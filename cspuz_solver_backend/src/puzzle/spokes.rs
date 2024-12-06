use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::spokes;

pub fn solve_spokes(url: &str) -> Result<Board, &'static str> {
    let clues = spokes::deserialize_problem(url).ok_or("invalid url")?;
    let (lines, lines_dr, lines_dl) = spokes::solve_spokes(&clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(
        BoardKind::Empty,
        height - 1,
        width - 1,
        is_unique(&(&lines, &lines_dr, &lines_dl)),
    );

    for y in 0..height {
        for x in 0..(width - 1) {
            match lines.horizontal[y][x] {
                Some(true) => {
                    board.push(Item {
                        y: y * 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: ItemKind::Wall,
                    });
                }
                Some(false) => (),
                None => {
                    board.push(Item {
                        y: y * 2,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::DottedWall,
                    });
                }
            }
        }
    }
    for y in 0..(height - 1) {
        for x in 0..width {
            match lines.vertical[y][x] {
                Some(true) => {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2,
                        color: "green",
                        kind: ItemKind::Wall,
                    });
                }
                Some(false) => (),
                None => {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2,
                        color: "black",
                        kind: ItemKind::DottedWall,
                    });
                }
            }
        }
    }
    for y in 0..(height - 1) {
        for x in 0..(width - 1) {
            match lines_dr[y][x] {
                Some(true) => {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 1,
                        color: "green",
                        kind: ItemKind::Backslash,
                    });
                }
                Some(false) => (),
                None => {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::DottedBackslash,
                    });
                }
            }

            match lines_dl[y][x] {
                Some(true) => {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 1,
                        color: "green",
                        kind: ItemKind::Slash,
                    });
                }
                Some(false) => (),
                None => {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::DottedSlash,
                    });
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = clues[y][x] {
                board.push(Item {
                    y: y * 2,
                    x: x * 2,
                    color: "white",
                    kind: ItemKind::FilledCircle,
                });
                board.push(Item {
                    y: y * 2,
                    x: x * 2,
                    color: "black",
                    kind: ItemKind::Circle,
                });

                if n >= 0 {
                    board.push(Item {
                        y: y * 2,
                        x: x * 2,
                        color: "black",
                        kind: ItemKind::Num(n),
                    });
                }
            }
        }
    }

    Ok(board)
}
