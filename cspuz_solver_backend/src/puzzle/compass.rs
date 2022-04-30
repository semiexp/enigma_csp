use crate::board::{Board, BoardKind, Compass, Item, ItemKind};
use cspuz_rs::puzzle::compass;

pub fn solve_compass(url: &str) -> Result<Board, &'static str> {
    let problem = compass::deserialize_problem(url).ok_or("invalid url")?;
    let ans = compass::solve_compass(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut data = vec![];
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "black",
                    ItemKind::Compass(Compass {
                        up: clue.up,
                        down: clue.down,
                        left: clue.left,
                        right: clue.right,
                    }),
                ));
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
                let mut need_default_edge = true;
                if let Some(b) = ans.horizontal[y][x] {
                    data.push(Item {
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
                    data.push(Item {
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
                    data.push(Item {
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
                    data.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "#cccccc",
                        kind: ItemKind::Wall,
                    });
                }
            }
        }
    }

    Ok(Board {
        kind: BoardKind::OuterGrid,
        height,
        width,
        data,
    })
}
