use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::graph;
use cspuz_rs::puzzle::yajilin_regions;

pub fn solve_yajilin_regions(url: &str) -> Result<Board, &'static str> {
    let (borders, clues) = yajilin_regions::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, is_black) =
        yajilin_regions::solve_yajilin_regions(&borders, &clues).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut board = Board::new(
        BoardKind::Grid,
        height,
        width,
        is_unique(&(&is_line, &is_black)),
    );

    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    let mut skip_line = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            row.push(is_black[y][x].unwrap_or(false));
        }
        skip_line.push(row);
    }
    board.add_lines_irrefutable_facts(&is_line, "green", Some(&skip_line));

    let rooms = graph::borders_to_rooms(&borders);
    assert_eq!(rooms.len(), clues.len());
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let (y, x) = rooms[i][0];
            board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
        }
    }

    Ok(board)
}
