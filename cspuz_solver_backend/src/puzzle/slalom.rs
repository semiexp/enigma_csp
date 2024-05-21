use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::slalom;

pub fn solve_slalom(url: &str) -> Result<Board, &'static str> {
    use slalom::{SlalomBlackCellDir, SlalomCell};

    let problem = slalom::deserialize_problem_as_primitive(url).ok_or("invalid url")?;
    let (is_black, gates, origin) = slalom::parse_primitive_problem(&problem);
    let is_line = slalom::solve_slalom(origin, &is_black, &gates).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    let (origin_y, origin_x) = origin;
    board.push(Item::cell(origin_y, origin_x, "black", ItemKind::Circle));
    board.push(Item::cell(
        origin_y,
        origin_x,
        "black",
        ItemKind::Num(gates.len() as i32),
    ));

    for y in 0..height {
        for x in 0..width {
            match problem.0[y][x] {
                SlalomCell::Black(d, n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                    if n >= 0 {
                        board.push(Item::cell(y, x, "white", ItemKind::Num(n)));
                    }
                    let arrow = match d {
                        SlalomBlackCellDir::Up => ItemKind::SideArrowUp,
                        SlalomBlackCellDir::Down => ItemKind::SideArrowDown,
                        SlalomBlackCellDir::Left => ItemKind::SideArrowLeft,
                        SlalomBlackCellDir::Right => ItemKind::SideArrowRight,
                        _ => continue,
                    };
                    board.push(Item::cell(y, x, "white", arrow));
                }
                SlalomCell::Horizontal => {
                    board.push(Item::cell(y, x, "black", ItemKind::DottedHorizontalWall));
                }
                SlalomCell::Vertical => {
                    board.push(Item::cell(y, x, "black", ItemKind::DottedVerticalWall));
                }
                SlalomCell::White => (),
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", Some(&is_black));

    Ok(board)
}
