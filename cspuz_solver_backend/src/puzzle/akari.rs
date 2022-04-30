use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::akari;

pub fn solve_akari(url: &str) -> Result<Board, &'static str> {
    let problem = akari::deserialize_problem(url).ok_or("invalid url")?;
    let ans = akari::solve_akari(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut data = vec![];
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                data.push(Item::cell(y, x, "black", ItemKind::Fill));
                if clue >= 0 {
                    data.push(Item::cell(y, x, "white", ItemKind::Num(clue)));
                }
            } else if let Some(a) = ans[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Circle } else { ItemKind::Dot },
                ));
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
