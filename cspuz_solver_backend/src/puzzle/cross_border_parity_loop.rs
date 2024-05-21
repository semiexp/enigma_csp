use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::cross_border_parity_loop::{self, CBPLCell};

pub fn solve_cross_border_parity_loop(url: &str) -> Result<Board, &'static str> {
    let (cells, clues_black, clues_white, borders) =
        cross_border_parity_loop::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, cell_state) = cross_border_parity_loop::solve_cross_border_parity_loop(
        &cells,
        &clues_black,
        &clues_white,
        &borders,
    )
    .ok_or("no answer")?;

    let height = cells.len();
    let width = cells[0].len();
    let mut board = Board::new(
        BoardKind::Grid,
        height,
        width,
        is_unique(&(&is_line, &cell_state)),
    );

    let mut is_skip = vec![vec![false; width]; height];
    for y in 0..height {
        for x in 0..width {
            if cells[y][x] == CBPLCell::Blocked {
                is_skip[y][x] = true;
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = cell_state[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    match n {
                        0 => "#cccccc",
                        1 => "#ffcccc",
                        2 => "#ccccff",
                        _ => unreachable!(),
                    },
                    ItemKind::Fill,
                ));
            }
            match cells[y][x] {
                CBPLCell::Empty => (),
                CBPLCell::Blocked => board.push(Item::cell(y, x, "black", ItemKind::Fill)),
                CBPLCell::BlackCircle => {
                    board.push(Item::cell(y, x, "#ff0000", ItemKind::SmallFilledCircle))
                }
                CBPLCell::WhiteCircle => {
                    board.push(Item::cell(y, x, "#0000ff", ItemKind::SmallCircle))
                }
            }

            if let Some(n) = clues_black[y][x] {
                board.push(Item::cell(y, x, "#ff0000", ItemKind::NumLowerRight(n)));
            }
            if let Some(n) = clues_white[y][x] {
                board.push(Item::cell(y, x, "#0000ff", ItemKind::NumUpperLeft(n)));
            }
        }
    }

    board.add_borders(&borders, "black");
    board.add_lines_irrefutable_facts(&is_line, "green", Some(&is_skip));

    Ok(board)
}
