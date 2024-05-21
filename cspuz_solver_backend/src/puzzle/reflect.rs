use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::reflect::{self, ReflectLinkClue};

pub fn solve_reflect_link(url: &str) -> Result<Board, &'static str> {
    let problem = reflect::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = reflect::solve_reflect_link(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    board.add_lines_irrefutable_facts(&is_line, "green", None);
    for y in 0..height {
        for x in 0..width {
            match &problem[y][x] {
                ReflectLinkClue::None => (),
                ReflectLinkClue::UpperLeft(n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::AboloUpperLeft));
                    if *n > 0 {
                        board.push(Item::cell(y, x, "white", ItemKind::NumUpperLeft(*n)));
                    }
                }
                ReflectLinkClue::UpperRight(n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::AboloUpperRight));
                    if *n > 0 {
                        board.push(Item::cell(y, x, "white", ItemKind::NumUpperRight(*n)));
                    }
                }
                ReflectLinkClue::LowerLeft(n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::AboloLowerLeft));
                    if *n > 0 {
                        board.push(Item::cell(y, x, "white", ItemKind::NumLowerLeft(*n)));
                    }
                }
                ReflectLinkClue::LowerRight(n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::AboloLowerRight));
                    if *n > 0 {
                        board.push(Item::cell(y, x, "white", ItemKind::NumLowerRight(*n)));
                    }
                }
                ReflectLinkClue::Cross => {
                    board.push(Item::cell(y, x, "black", ItemKind::Plus));
                }
            }
        }
    }

    Ok(board)
}
