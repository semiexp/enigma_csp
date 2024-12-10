use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::items::Arrow;
use cspuz_rs::puzzle::sniping_arrow;

fn convert_arrow(arrow: Arrow) -> ItemKind {
    match arrow {
        Arrow::Up => ItemKind::ArrowUp,
        Arrow::Down => ItemKind::ArrowDown,
        Arrow::Left => ItemKind::ArrowLeft,
        Arrow::Right => ItemKind::ArrowRight,
        Arrow::Unspecified => ItemKind::Dot,
    }
}

pub fn solve_sniping_arrow(url: &str) -> Result<Board, &'static str> {
    let clues = sniping_arrow::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, arrow) = sniping_arrow::solve_sniping_arrow(&clues).ok_or("no answer")?;

    let height = clues.len();
    let width = clues[0].len();
    let mut board = Board::new(
        BoardKind::Grid,
        height,
        width,
        is_unique(&(&is_line, &arrow)),
    );

    for y in 0..height {
        for x in 0..width {
            if let Some((n, ar)) = clues[y][x] {
                if let Some(n) = n {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                }
                if let Some(ar) = ar {
                    board.push(Item::cell(y, x, "black", convert_arrow(ar)));
                    continue;
                }

                if let Some(ar) = arrow[y][x] {
                    board.push(Item::cell(y, x, "green", convert_arrow(ar)));
                }
            } else {
                board.push(Item::cell(y, x, "black", ItemKind::Fill));
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", None);

    Ok(board)
}
