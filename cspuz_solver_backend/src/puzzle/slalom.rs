use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::slalom;

pub fn solve_slalom(url: &str) -> Result<Board, &'static str> {
    use slalom::{SlalomBlackCellDir, SlalomCell};

    let problem = slalom::deserialize_problem_as_primitive(url).ok_or("invalid url")?;
    let (is_black, gates, origin) = slalom::parse_primitive_problem(&problem);
    let is_line = slalom::solve_slalom(origin, &is_black, &gates).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut data = vec![];

    let (origin_y, origin_x) = origin;
    data.push(Item::cell(origin_y, origin_x, "black", ItemKind::Circle));
    data.push(Item::cell(
        origin_y,
        origin_x,
        "black",
        ItemKind::Num(gates.len() as i32),
    ));

    for y in 0..height {
        for x in 0..width {
            match problem.0[y][x] {
                SlalomCell::Black(d, n) => {
                    data.push(Item::cell(y, x, "black", ItemKind::Fill));
                    if n >= 0 {
                        data.push(Item::cell(y, x, "white", ItemKind::Num(n)));
                    }
                    let arrow = match d {
                        SlalomBlackCellDir::Up => ItemKind::SideArrowUp,
                        SlalomBlackCellDir::Down => ItemKind::SideArrowDown,
                        SlalomBlackCellDir::Left => ItemKind::SideArrowLeft,
                        SlalomBlackCellDir::Right => ItemKind::SideArrowRight,
                        _ => continue,
                    };
                    data.push(Item::cell(y, x, "white", arrow));
                }
                SlalomCell::Horizontal => {
                    data.push(Item::cell(y, x, "black", ItemKind::DottedHorizontalWall));
                }
                SlalomCell::Vertical => {
                    data.push(Item::cell(y, x, "black", ItemKind::DottedVerticalWall));
                }
                SlalomCell::White => (),
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && !(is_black[y][x] || is_black[y + 1][x]) {
                if let Some(b) = is_line.vertical[y][x] {
                    data.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
            if x < width - 1 && !(is_black[y][x] || is_black[y][x + 1]) {
                if let Some(b) = is_line.horizontal[y][x] {
                    data.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
        }
    }

    Ok(Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    })
}
