use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::firewalk;

pub fn solve_firewalk(url: &str) -> Result<Board, &'static str> {
    let (fire_cell, num) = firewalk::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, fire_cell_mode) =
        firewalk::solve_firewalk(&fire_cell, &num).ok_or("no answer")?;

    let height = fire_cell.len();
    let width = fire_cell[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    for y in 0..height {
        for x in 0..width {
            if fire_cell[y][x] {
                board.push(Item::cell(y, x, "#ffe0e0", ItemKind::Fill));
            }
        }
    }
    board.add_lines_irrefutable_facts(&is_line, "green", None);

    for y in 0..height {
        for x in 0..width {
            if fire_cell[y][x] {
                let cell_item = match fire_cell_mode[y][x] {
                    Some(true) => {
                        let ur = y > 0 && is_line.vertical[y - 1][x] == Some(true);
                        let dl = x > 0 && is_line.horizontal[y][x - 1] == Some(true);

                        match (ur, dl) {
                            (true, true) => Some(ItemKind::FirewalkCellUrDl),
                            (true, false) => Some(ItemKind::FirewalkCellUr),
                            (false, true) => Some(ItemKind::FirewalkCellDl),
                            (false, false) => None,
                        }
                    }
                    Some(false) => {
                        let ul = y > 0 && is_line.vertical[y - 1][x] == Some(true);
                        let dr = x < width - 1 && is_line.horizontal[y][x] == Some(true);

                        match (ul, dr) {
                            (true, true) => Some(ItemKind::FirewalkCellUlDr),
                            (true, false) => Some(ItemKind::FirewalkCellUl),
                            (false, true) => Some(ItemKind::FirewalkCellDr),
                            (false, false) => None,
                        }
                    }
                    None => Some(ItemKind::FirewalkCellUnknown),
                };

                if let Some(cell_item) = cell_item {
                    board.push(Item::cell(y, x, "green", cell_item));
                }
            }
            if let Some(n) = num[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            }
        }
    }

    Ok(board)
}
