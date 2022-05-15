use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::tapa;

pub fn solve_tapa(url: &str) -> Result<Board, &'static str> {
    let problem = tapa::deserialize_problem(url).ok_or("invalid url")?;
    let ans = tapa::solve_tapa(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::TapaClue(clue)));
            } else if let Some(a) = ans[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(board)
}
