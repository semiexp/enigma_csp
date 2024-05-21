use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::square_jam;

pub fn solve_square_jam(url: &str) -> Result<Board, &'static str> {
    let problem = square_jam::deserialize_problem(url).ok_or("invalid url")?;
    let border = square_jam::solve_square_jam(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::OuterGrid, height, width, is_unique(&border));

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                if n >= 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                } else if n == -2 {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
                let mut need_default_edge = true;
                if let Some(b) = border.horizontal[y][x] {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: if b {
                            ItemKind::BoldWall
                        } else {
                            ItemKind::Cross
                        },
                    });
                    if b {
                        need_default_edge = false;
                    }
                }
                if need_default_edge {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "#cccccc",
                        kind: ItemKind::Wall,
                    });
                }
            }
            if x < width - 1 {
                let mut need_default_edge = true;
                if let Some(b) = border.vertical[y][x] {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "green",
                        kind: if b {
                            ItemKind::BoldWall
                        } else {
                            ItemKind::Cross
                        },
                    });
                    if b {
                        need_default_edge = false;
                    }
                }
                if need_default_edge {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "#cccccc",
                        kind: ItemKind::Wall,
                    });
                }
            }
        }
    }

    Ok(board)
}
