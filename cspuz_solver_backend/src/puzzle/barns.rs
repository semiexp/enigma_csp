use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::barns;

pub fn solve_barns(url: &str) -> Result<Board, &'static str> {
    let (icebarn, borders) = barns::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = barns::solve_barns(&icebarn, &borders).ok_or("no answer")?;

    let height = icebarn.len();
    let width = icebarn[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    board.add_borders(&borders, "black");
    for y in 0..height {
        for x in 0..width {
            if icebarn[y][x] {
                board.push(Item::cell(y, x, "#e0e0ff", ItemKind::Fill));
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
                if !borders.horizontal[y][x] {
                    if let Some(b) = is_line.vertical[y][x] {
                        board.push(Item {
                            y: y * 2 + 2,
                            x: x * 2 + 1,
                            color: "green",
                            kind: if b { ItemKind::Line } else { ItemKind::Cross },
                        });
                    }
                }
            }
            if x < width - 1 {
                if !borders.vertical[y][x] {
                    if let Some(b) = is_line.horizontal[y][x] {
                        board.push(Item {
                            y: y * 2 + 1,
                            x: x * 2 + 2,
                            color: "green",
                            kind: if b { ItemKind::Line } else { ItemKind::Cross },
                        });
                    }
                }
            }
        }
    }

    Ok(board)
}
