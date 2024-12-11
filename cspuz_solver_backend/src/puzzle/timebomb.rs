use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::timebomb;

pub fn solve_timebomb(url: &str) -> Result<Board, &'static str> {
    let problem = timebomb::deserialize_problem(url).ok_or("invalid url")?;
    let (has_number, num) = timebomb::solve_timebomb(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(
        BoardKind::Grid,
        height,
        width,
        is_unique(&(&has_number, &num)),
    );
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                if clue >= 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Circle));
                    board.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                } else if clue == -1 {
                    board.push(Item::cell(y, x, "black", ItemKind::Circle));
                    board.push(Item::cell(y, x, "black", ItemKind::Text("?")));
                } else if clue == -2 {
                    board.push(Item::cell(y, x, "black", ItemKind::FilledCircle));
                }
            } else if let Some(n) = num[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if n == -1 {
                        ItemKind::Dot
                    } else {
                        ItemKind::Num(n)
                    },
                ));
            } else if has_number[y][x] == Some(true) {
                board.push(Item::cell(y, x, "green", ItemKind::Text("?")));
            }
        }
    }

    Ok(board)
}
