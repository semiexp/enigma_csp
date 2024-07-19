use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::evolmino;

pub fn solve_evolmino(url: &str) -> Result<Board, &'static str> {
    let problem = evolmino::deserialize_problem(url).ok_or("invalid url")?;
    let is_square = evolmino::solve_evolmino(&problem).ok_or("no answer")?;

    let height = problem.cells.len();
    let width = problem.cells[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_square));

    for arrow in &problem.arrows {
        for i in 1..arrow.len() {
            let (y1, x1) = arrow[i - 1];
            let (y2, x2) = arrow[i];

            board.push(Item {
                y: y1 + y2 + 1,
                x: x1 + x2 + 1,
                color: "black",
                kind: ItemKind::Line,
            });
        }
    }

    for y in 0..height {
        for x in 0..width {
            if problem.cells[y][x] == evolmino::ProblemCell::Black {
                board.push(Item::cell(y, x, "black", ItemKind::Fill));
                continue;
            }
            if problem.cells[y][x] == evolmino::ProblemCell::Square {
                board.push(Item::cell(y, x, "black", ItemKind::Square));
                continue;
            }
            match is_square[y][x] {
                Some(true) => {
                    board.push(Item::cell(y, x, "green", ItemKind::Square));
                }
                Some(false) => {
                    board.push(Item::cell(y, x, "green", ItemKind::Dot));
                }
                None => (),
            }
        }
    }

    Ok(board)
}
