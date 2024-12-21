use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::slicy;

pub fn solve_slicy(url: &str) -> Result<Board, &'static str> {
    let borders = slicy::deserialize_problem(url).ok_or("invalid url")?;
    let ans = slicy::solve_slicy(&borders).ok_or("no answer")?;

    let (a, b, c, d) = ans.dims();
    let mut board = Board::new(
        BoardKind::Empty,
        (a + c - 1) * 2,
        a + b * 2 + d - 2,
        is_unique(&ans.flatten().to_vec()),
    );

    for &(y, x) in ans.cells() {
        let ty = y * 2;
        let tx = if y >= a - 1 {
            x * 2 - (y - (a - 1))
        } else {
            x * 2 + (a - 1 - y)
        };

        for dy in 0..2 {
            for dx in 0..2 {
                if let Some(b) = ans[(y, x)] {
                    board.push(Item {
                        y: ty * 2 + 1 + dy * 2,
                        x: tx * 2 + 1 + dx * 2,
                        color: if b { "green" } else { "#cccccc" },
                        kind: ItemKind::Fill,
                    });
                }
            }
        }
    }

    for &(y, x) in ans.cells() {
        let ty = y * 2;
        let tx = if y >= a - 1 {
            x * 2 - (y - (a - 1))
        } else {
            x * 2 + (a - 1 - y)
        };

        board.push(Item {
            y: ty * 2,
            x: tx * 2 + 1,
            color: "black",
            kind: if !ans.is_valid_coord_offset((y, x), (-1, -1))
                || borders.to_bottom_right[(y - 1, x - 1)]
            {
                ItemKind::BoldWall
            } else {
                ItemKind::DottedWall
            },
        });
        board.push(Item {
            y: ty * 2 + 4,
            x: tx * 2 + 3,
            color: "black",
            kind: if !ans.is_valid_coord_offset((y, x), (1, 1)) || borders.to_bottom_right[(y, x)] {
                ItemKind::BoldWall
            } else {
                ItemKind::DottedWall
            },
        });
        board.push(Item {
            y: ty * 2,
            x: tx * 2 + 3,
            color: "black",
            kind: if !ans.is_valid_coord_offset((y, x), (-1, 0))
                || borders.to_bottom_left[(y - 1, x)]
            {
                ItemKind::BoldWall
            } else {
                ItemKind::DottedWall
            },
        });
        board.push(Item {
            y: ty * 2 + 4,
            x: tx * 2 + 1,
            color: "black",
            kind: if !ans.is_valid_coord_offset((y, x), (1, 0)) || borders.to_bottom_left[(y, x)] {
                ItemKind::BoldWall
            } else {
                ItemKind::DottedWall
            },
        });
        for t in [1, 3] {
            board.push(Item {
                y: ty * 2 + t,
                x: tx * 2,
                color: "black",
                kind: if !ans.is_valid_coord_offset((y, x), (0, -1)) || borders.to_right[(y, x - 1)]
                {
                    ItemKind::BoldWall
                } else {
                    ItemKind::DottedWall
                },
            });
            board.push(Item {
                y: ty * 2 + t,
                x: tx * 2 + 4,
                color: "black",
                kind: if !ans.is_valid_coord_offset((y, x), (0, 1)) || borders.to_right[(y, x)] {
                    ItemKind::BoldWall
                } else {
                    ItemKind::DottedWall
                },
            });
        }
    }

    Ok(board)
}
