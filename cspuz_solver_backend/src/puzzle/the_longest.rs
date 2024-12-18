use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::the_longest;

pub fn solve_the_longest(url: &str) -> Result<Board, &'static str> {
    let problem = the_longest::deserialize_problem(url).ok_or("invalid url")?;
    let ans = the_longest::solve_the_longest(&problem).ok_or("no answer")?;

    let height = ans.vertical.len();
    let width = ans.vertical[0].len() + 1;
    let mut board = Board::new(BoardKind::DotGrid, height, width, is_unique(&ans));

    for y in 0..=height {
        for x in 0..width {
            if problem.horizontal[y][x] {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
                continue;
            }

            if y == 0 || y == height {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: ItemKind::Wall,
                });
                continue;
            }

            if let Some(b) = ans.horizontal[y - 1][x] {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                });
            }
        }
    }
    for y in 0..height {
        for x in 0..=width {
            if problem.vertical[y][x] {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
                continue;
            }

            if x == 0 || x == width {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: ItemKind::Wall,
                });
                continue;
            }

            if let Some(b) = ans.vertical[y][x - 1] {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                });
            }
        }
    }

    Ok(board)
}
