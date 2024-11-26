use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::akari_rgb::{self, AkariRGBClue};

pub fn solve_akari_rgb(url: &str) -> Result<Board, &'static str> {
    let problem = akari_rgb::deserialize_problem(url).ok_or("invalid url")?;
    let ans = akari_rgb::solve_akari_rgb(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&ans));
    for y in 0..height {
        for x in 0..width {
            match problem[y][x] {
                AkariRGBClue::Block => {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                    continue;
                }
                AkariRGBClue::Num(n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                    board.push(Item::cell(y, x, "white", ItemKind::Num(n)));
                    continue;
                }
                AkariRGBClue::Empty => (),
                AkariRGBClue::R => {
                    board.push(Item::cell(y, x, "#ffcccc", ItemKind::Fill));
                }
                AkariRGBClue::G => {
                    board.push(Item::cell(y, x, "#ccffcc", ItemKind::Fill));
                }
                AkariRGBClue::B => {
                    board.push(Item::cell(y, x, "#ccccff", ItemKind::Fill));
                }
                AkariRGBClue::RG => {
                    board.push(Item::cell(y, x, "#ffffcc", ItemKind::Fill));
                }
                AkariRGBClue::GB => {
                    board.push(Item::cell(y, x, "#ccffff", ItemKind::Fill));
                }
                AkariRGBClue::BR => {
                    board.push(Item::cell(y, x, "#ffccff", ItemKind::Fill));
                }
            }

            match ans[y][x] {
                None => (),
                Some(0) => {
                    board.push(Item::cell(y, x, "black", ItemKind::Dot));
                }
                Some(1) => {
                    board.push(Item::cell(y, x, "#ff0000", ItemKind::FilledCircle));
                    board.push(Item::cell(y, x, "white", ItemKind::Text("R")));
                }
                Some(2) => {
                    board.push(Item::cell(y, x, "#00ff00", ItemKind::FilledCircle));
                    board.push(Item::cell(y, x, "white", ItemKind::Text("G")));
                }
                Some(3) => {
                    board.push(Item::cell(y, x, "#0000ff", ItemKind::FilledCircle));
                    board.push(Item::cell(y, x, "white", ItemKind::Text("B")));
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(board)
}
