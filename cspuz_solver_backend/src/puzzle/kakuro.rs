use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::kakuro;

pub fn solve_kakuro(url: &str) -> Result<Board, &'static str> {
    let problem = kakuro::deserialize_problem(url).ok_or("invalid url")?;
    let answer: Vec<Vec<Option<i32>>> = kakuro::solve_kakuro(&problem).ok_or("no answer")?;

    let height = answer.len();
    let width = answer[0].len();
    let mut board = Board::new(BoardKind::OuterGrid, height, width, is_unique(&answer));

    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                board.push(Item::cell(y, x, "#cccccc", ItemKind::Fill));
                board.push(Item::cell(y, x, "black", ItemKind::Backslash));

                if let Some(n) = clue.down {
                    if n > 0 {
                        board.push(Item::cell(y, x, "black", ItemKind::NumLowerLeft(n)));
                    }
                }
                if let Some(n) = clue.right {
                    if n > 0 {
                        board.push(Item::cell(y, x, "black", ItemKind::NumUpperRight(n)));
                    }
                }
            } else {
                if let Some(n) = answer[y][x] {
                    board.push(Item::cell(y, x, "green", ItemKind::Num(n)));
                }
            }
        }
    }

    Ok(board)
}
