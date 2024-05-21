use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::hashi;

pub fn solve_hashi(url: &str) -> Result<Board, &'static str> {
    let clues = hashi::deserialize_problem(url).ok_or("invalid url")?;
    let num_line = hashi::solve_hashi(&clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(BoardKind::Empty, height, width, is_unique(&num_line));

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
                if let Some(n) = num_line.vertical[y][x] {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: match n {
                            0 => ItemKind::Cross,
                            1 => ItemKind::Line,
                            2 => ItemKind::DoubleLine,
                            _ => unreachable!(),
                        },
                    });
                }
            }
            if x < width - 1 {
                if let Some(n) = num_line.horizontal[y][x] {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "green",
                        kind: match n {
                            0 => ItemKind::Cross,
                            1 => ItemKind::Line,
                            2 => ItemKind::DoubleLine,
                            _ => unreachable!(),
                        },
                    });
                }
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(n) = clues[y][x] {
                board.push(Item::cell(y, x, "white", ItemKind::FilledCircle));
                board.push(Item::cell(y, x, "black", ItemKind::Circle));
                if n > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                }
            }
        }
    }

    Ok(board)
}
