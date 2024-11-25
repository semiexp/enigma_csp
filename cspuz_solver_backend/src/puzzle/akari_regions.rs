use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::akari_regions;

pub fn solve_akari_regions(url: &str) -> Result<Board, &'static str> {
    let (borders, clues, has_block) =
        akari_regions::deserialize_problem(url).ok_or("invalid url")?;
    let has_light =
        akari_regions::solve_akari_region(&borders, &clues, &has_block).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&has_light));

    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if has_block[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Fill));
                continue;
            }
            if let Some(b) = has_light[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b {
                        ItemKind::FilledCircle
                    } else {
                        ItemKind::Dot
                    },
                ));
            }
            if let Some(n) = clues[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::NumUpperLeft(n)));
            }
        }
    }

    Ok(board)
}
