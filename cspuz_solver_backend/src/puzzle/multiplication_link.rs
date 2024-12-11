use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::multiplication_link;

pub fn solve_multiplication_link(url: &str) -> Result<Board, &'static str> {
    let problem = multiplication_link::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = multiplication_link::solve_multiplication_link(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    let mut skip_line = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            row.push(problem[y][x] == Some(-2));
        }
        skip_line.push(row);
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "black",
                    if n == -2 {
                        ItemKind::Fill
                    } else if n == -1 {
                        ItemKind::Circle
                    } else {
                        ItemKind::Num(n)
                    },
                ));
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", Some(&skip_line));

    Ok(board)
}
