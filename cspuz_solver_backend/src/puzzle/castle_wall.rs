use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::castle_wall;

pub fn solve_castle_wall(url: &str) -> Result<Board, &'static str> {
    use castle_wall::Side;
    use cspuz_rs::items::NumberedArrow;
    let problem = castle_wall::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = castle_wall::solve_castle_wall(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    let mut skip_line = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            row.push(problem[y][x].is_some());
        }
        skip_line.push(row);
    }
    for y in 0..height {
        for x in 0..width {
            if let Some((side, arrow)) = problem[y][x] {
                let arrow_color = match side {
                    Side::Outside => "white",
                    _ => "black",
                };
                if side == Side::Outside {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                } else if side == Side::Unspecified {
                    board.push(Item::cell(y, x, "#cccccc", ItemKind::Fill));
                }
                let (arrow, n) = match arrow {
                    NumberedArrow::Unspecified(n) => (None, n),
                    NumberedArrow::Up(n) => (Some(ItemKind::SideArrowUp), n),
                    NumberedArrow::Down(n) => (Some(ItemKind::SideArrowDown), n),
                    NumberedArrow::Left(n) => (Some(ItemKind::SideArrowLeft), n),
                    NumberedArrow::Right(n) => (Some(ItemKind::SideArrowRight), n),
                };
                if n >= 0 {
                    if let Some(arrow) = arrow {
                        board.push(Item::cell(y, x, arrow_color, arrow));
                    }
                    board.push(Item::cell(y, x, arrow_color, ItemKind::Num(n)));
                }
            }
        }
    }

    board.add_lines_irrefutable_facts(&is_line, "green", Some(&skip_line));

    Ok(board)
}
