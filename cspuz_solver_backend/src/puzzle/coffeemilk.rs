use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::coffeemilk;

pub fn solve_coffeemilk(url: &str) -> Result<Board, &'static str> {
    let clues = coffeemilk::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = coffeemilk::solve_coffeemilk(&clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(BoardKind::Empty, height, width, is_unique(&is_line));

    board.add_lines_irrefutable_facts(&is_line, "green", None);

    for y in 0..height {
        for x in 0..width {
            if clues[y][x] == 1 {
                board.push(Item::cell(y, x, "white", ItemKind::FilledCircle));
                board.push(Item::cell(y, x, "black", ItemKind::Circle));
            } else if clues[y][x] == 2 {
                board.push(Item::cell(y, x, "black", ItemKind::FilledCircle));
            } else if clues[y][x] == 3 {
                board.push(Item::cell(y, x, "#cccccc", ItemKind::FilledCircle));
                board.push(Item::cell(y, x, "black", ItemKind::Circle));
            }
        }
    }

    Ok(board)
}
