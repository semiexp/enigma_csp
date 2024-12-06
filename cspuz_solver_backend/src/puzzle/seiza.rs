use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::seiza;

pub fn solve_seiza(url: &str) -> Result<Board, &'static str> {
    let (absent_cell, num, borders) = seiza::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, is_star) = seiza::solve_seiza(&absent_cell, &num, &borders).ok_or("no answer")?;

    let height = absent_cell.len();
    let width = absent_cell[0].len();
    let mut board = Board::new(
        BoardKind::Grid,
        height,
        width,
        is_unique(&(&is_line, &is_star)),
    );

    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if absent_cell[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Fill));
                continue;
            }
            if let Some(b) = is_star[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b {
                        ItemKind::FilledCircle
                    } else {
                        ItemKind::Dot
                    },
                ));
            }
            if let Some(n) = num[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::NumUpperLeft(n)));
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", Some(&absent_cell));

    Ok(board)
}
