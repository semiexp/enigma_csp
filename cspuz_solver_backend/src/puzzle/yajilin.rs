use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::yajilin;

pub fn solve_yajilin(url: &str) -> Result<Board, &'static str> {
    use yajilin::YajilinClue;

    let problem = yajilin::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, is_black) = yajilin::solve_yajilin(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut data = vec![];

    let mut skip_line = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            row.push(problem[y][x].is_some() || is_black[y][x] == Some(true));
        }
        skip_line.push(row);
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                let (arrow, n) = match clue {
                    YajilinClue::Unspecified(n) => (None, n),
                    YajilinClue::Up(n) => (Some(ItemKind::SideArrowUp), n),
                    YajilinClue::Down(n) => (Some(ItemKind::SideArrowDown), n),
                    YajilinClue::Left(n) => (Some(ItemKind::SideArrowLeft), n),
                    YajilinClue::Right(n) => (Some(ItemKind::SideArrowRight), n),
                };
                if let Some(arrow) = arrow {
                    data.push(Item::cell(y, x, "black", arrow));
                }
                data.push(Item::cell(
                    y,
                    x,
                    "black",
                    if n >= 0 {
                        ItemKind::Num(n)
                    } else {
                        ItemKind::Text("?")
                    },
                ));
            } else if let Some(b) = is_black[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && !(skip_line[y][x] || skip_line[y + 1][x]) {
                if let Some(b) = is_line.vertical[y][x] {
                    data.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
            if x < width - 1 && !(skip_line[y][x] || skip_line[y][x + 1]) {
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
