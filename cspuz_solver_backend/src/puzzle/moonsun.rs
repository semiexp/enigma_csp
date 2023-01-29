use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::moonsun;

pub fn solve_moonsun(url: &str) -> Result<Board, &'static str> {
    let (borders, clues) = moonsun::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = moonsun::solve_moonsun(&borders, &clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);
    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if clues[y][x] != 0 {
                board.push(Item::cell(
                    y,
                    x,
                    "black",
                    if clues[y][x] == 1 {
                        ItemKind::Circle
                    } else {
                        ItemKind::FilledCircle
                    },
                ))
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", None);

    Ok(board)
}
