use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::yajilin;

pub fn solve_yajilin(url: &str) -> Result<Board, &'static str> {
    use yajilin::YajilinClue;

    let problem = yajilin::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, is_black) = yajilin::solve_yajilin(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

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
                    board.push(Item::cell(y, x, "black", arrow));
                }
                board.push(Item::cell(
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
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", Some(&skip_line));

    Ok(board)
}
