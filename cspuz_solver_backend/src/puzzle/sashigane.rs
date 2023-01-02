use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::sasahigane::{self, SashiganeClue};

pub fn solve_sashigane(url: &str) -> Result<Board, &'static str> {
    let problem = sasahigane::deserialize_problem(url).ok_or("invalid url")?;
    let ans = sasahigane::solve_sashigane(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::OuterGrid, height, width);
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                let kind = match clue {
                    SashiganeClue::Up => ItemKind::ArrowUp,
                    SashiganeClue::Down => ItemKind::ArrowDown,
                    SashiganeClue::Left => ItemKind::ArrowLeft,
                    SashiganeClue::Right => ItemKind::ArrowRight,
                    SashiganeClue::Corner(_) => ItemKind::Circle,
                };
                board.push(Item::cell(y, x, "black", kind));
                if let SashiganeClue::Corner(n) = clue {
                    if n > 0 {
                        board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                    }
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
