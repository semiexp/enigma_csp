use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::icewalk;

pub fn solve_icewalk(url: &str) -> Result<Board, &'static str> {
    let (icebarn, num) = icewalk::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = icewalk::solve_icewalk(&icebarn, &num).ok_or("no answer")?;

    let height = icebarn.len();
    let width = icebarn[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    for y in 0..height {
        for x in 0..width {
            if icebarn[y][x] {
                board.push(Item::cell(y, x, "#e0e0ff", ItemKind::Fill));
            }
            if let Some(n) = num[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", None);

    Ok(board)
}
