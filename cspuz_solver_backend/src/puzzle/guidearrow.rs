use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::guidearrow::{self, GuidearrowClue};

pub fn solve_guidearrow(url: &str) -> Result<Board, &'static str> {
    let (ty, tx, clues) = guidearrow::deserialize_problem(url).ok_or("invalid url")?;
    let ans = guidearrow::solve_guidearrow(ty, tx, &clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);
    board.push(Item::cell(
        ty as usize,
        tx as usize,
        "black",
        ItemKind::Circle,
    ));
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = clues[y][x] {
                let kind = match clue {
                    GuidearrowClue::Up => ItemKind::ArrowUp,
                    GuidearrowClue::Down => ItemKind::ArrowDown,
                    GuidearrowClue::Left => ItemKind::ArrowLeft,
                    GuidearrowClue::Right => ItemKind::ArrowRight,
                    GuidearrowClue::Unknown => ItemKind::Text("?"),
                };
                board.push(Item::cell(y, x, "black", kind));
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if clues[y][x].is_some() || (y, x) == (ty, tx) {
                continue;
            }
            if let Some(b) = ans[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Fill } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(board)
}
