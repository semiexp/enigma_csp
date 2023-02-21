use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::tricklayer;

pub fn solve_tricklayer(url: &str) -> Result<Board, &'static str> {
    let problem = tricklayer::deserialize_problem(url).ok_or("invalid url")?;
    let ans = tricklayer::solve_tricklayer(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::OuterGrid, height, width);
    for y in 0..height {
        for x in 0..width {
            if problem[y][x] {
                board.push(Item::cell(y, x, "#cccccc", ItemKind::Fill));
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && !problem[y][x] && !problem[y + 1][x] {
                let mut need_default_edge = true;
                if let Some(b) = ans.horizontal[y][x] {
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
            if x < width - 1 && !problem[y][x] && !problem[y][x + 1] {
                let mut need_default_edge = true;
                if let Some(b) = ans.vertical[y][x] {
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
