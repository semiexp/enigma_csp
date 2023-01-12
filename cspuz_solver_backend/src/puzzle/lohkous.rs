use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::lohkous;

pub fn solve_lohkous(url: &str) -> Result<Board, &'static str> {
    let problem = lohkous::deserialize_problem(url).ok_or("invalid url")?;
    let ans = lohkous::solve_lohkous(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::OuterGrid, height, width);
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = &problem[y][x] {
                if clue.len() <= 4 {
                    let mut c = [-1, -1, -1, -1];
                    for i in 0..clue.len() {
                        c[i] = if clue[i] == -1 { -2 } else { clue[i] };
                    }
                    board.push(Item::cell(y, x, "black", ItemKind::TapaClue(c)));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::Text("...")));
                }
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
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
            if x < width - 1 {
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
