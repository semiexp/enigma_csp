use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::parrot_loop;

pub fn solve_parrot_loop(url: &str) -> Result<Board, &'static str> {
    let problem = parrot_loop::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = parrot_loop::solve_parrot_loop(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    let mut skip_line = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            row.push(problem[y][x] == Some(-1));
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
                    if n < 0 {
                        ItemKind::Fill
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
