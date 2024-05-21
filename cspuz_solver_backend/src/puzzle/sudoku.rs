use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::Uniqueness;
use cspuz_rs::puzzle::sudoku;

pub fn solve_sudoku(url: &str) -> Result<Board, &'static str> {
    let problem = sudoku::deserialize_problem(url).ok_or("invalid url")?;
    let ans = sudoku::solve_sudoku_as_cands(&problem).ok_or("no answer")?;

    let height = ans.len();
    let width = ans[0].len();

    let mut is_unique = Uniqueness::Unique;
    for y in 0..height {
        for x in 0..width {
            if ans[y][x].iter().filter(|&&b| b).count() != 1 {
                is_unique = Uniqueness::NonUnique;
            }
        }
    }
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique);

    let (bh, bw) = match height {
        4 => (2, 2),
        6 => (2, 3),
        9 => (3, 3),
        16 => (4, 4),
        25 => (5, 5),
        _ => return Err("invalid size"),
    };

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            } else {
                let mut cands = vec![];
                for i in 0..height {
                    if ans[y][x][i] {
                        cands.push(i as i32 + 1);
                    }
                }
                if cands.len() == 1 {
                    board.push(Item::cell(y, x, "green", ItemKind::Num(cands[0])));
                } else {
                    board.push(Item::cell(
                        y,
                        x,
                        "green",
                        ItemKind::SudokuCandidateSet(bw as i32, cands),
                    ));
                }
            }
        }
    }
    for x in 0..bh {
        for y in 0..height {
            board.push(Item {
                y: 2 * y + 1,
                x: 2 * x * bw,
                color: "black",
                kind: ItemKind::BoldWall,
            });
        }
    }
    for y in 0..bw {
        for x in 0..width {
            board.push(Item {
                y: 2 * y * bh,
                x: 2 * x + 1,
                color: "black",
                kind: ItemKind::BoldWall,
            });
        }
    }

    Ok(board)
}
