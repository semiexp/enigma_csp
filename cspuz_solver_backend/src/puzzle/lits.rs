use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::lits;

pub fn solve_lits(url: &str) -> Result<Board, &'static str> {
    let borders = lits::deserialize_problem(url).ok_or("invalid url")?;
    let is_black = lits::solve_lits(&borders).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut data = vec![];

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && borders.horizontal[y][x] {
                data.push(Item {
                    y: y * 2 + 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
            if x < width - 1 && borders.vertical[y][x] {
                data.push(Item {
                    y: y * 2 + 1,
                    x: x * 2 + 2,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    })
}
