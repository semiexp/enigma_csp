use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::Uniqueness;
use cspuz_rs::puzzle::kouchoku;

pub fn solve_kouchoku(url: &str) -> Result<Board, &'static str> {
    let problem = kouchoku::deserialize_problem(url).ok_or("invalid url")?;
    let (fixed_lines, undet_lines) = kouchoku::solve_kouchoku(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(
        BoardKind::Empty,
        height,
        width,
        if undet_lines.len() == 0 {
            Uniqueness::Unique
        } else {
            Uniqueness::NonUnique
        },
    );
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 {
                board.push(Item {
                    y: y * 2 + 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: ItemKind::DottedLine,
                });
            }
            if x < width - 1 {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2 + 2,
                    color: "black",
                    kind: ItemKind::DottedLine,
                });
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                if clue >= 0 {
                    board.push(Item::cell(y, x, "white", ItemKind::FilledCircle));
                    board.push(Item::cell(y, x, "black", ItemKind::Circle));
                    board.push(Item::cell(y, x, "black", ItemKind::Num(clue + 1)));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::FilledCircle));
                }
            }
        }
    }

    for ((x1, y1), (x2, y2)) in fixed_lines {
        board.push(Item {
            y: y1 * 2 + 1,
            x: x1 * 2 + 1,
            color: "green",
            kind: ItemKind::LineTo(y2 as i32 * 2 + 1, x2 as i32 * 2 + 1),
        });
    }
    for ((x1, y1), (x2, y2)) in undet_lines {
        board.push(Item {
            y: y1 * 2 + 1,
            x: x1 * 2 + 1,
            color: "#888888",
            kind: ItemKind::LineTo(y2 as i32 * 2 + 1, x2 as i32 * 2 + 1),
        });
    }

    Ok(board)
}
