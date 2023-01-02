use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::sudoku;

pub fn solve_sudoku(url: &str) -> Result<Board, &'static str> {
    let problem = sudoku::deserialize_problem(url).ok_or("invalid url")?;
    let ans = sudoku::solve_sudoku(&problem).ok_or("no answer")?;

    let height = ans.len();
    let width = ans[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            } else if let Some(n) = ans[y][x] {
                board.push(Item::cell(y, x, "green", ItemKind::Num(n)));
            }
        }
    }
    let mut s = None;
    for i in 2..=5 {
        if i * i == height {
            s = Some(i);
        }
    }
    if let Some(s) = s {
        for x in 0..s {
            for y in 0..height {
                board.push(Item {
                    y: 2 * y + 1,
                    x: 2 * x * s,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
        }
        for y in 0..s {
            for x in 0..width {
                board.push(Item {
                    y: 2 * y * s,
                    x: 2 * x + 1,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
        }
    }

    Ok(board)
}
