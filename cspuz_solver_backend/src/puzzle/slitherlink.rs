use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::slitherlink;

pub fn solve_slitherlink(url: &str) -> Result<Board, &'static str> {
    let problem = slitherlink::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = slitherlink::solve_slitherlink(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::DotGrid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            }
        }
    }
    for y in 0..height {
        for x in 0..=width {
            if let Some(b) = is_line.vertical[y][x] {
                board.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }
    for y in 0..=height {
        for x in 0..width {
            if let Some(b) = is_line.horizontal[y][x] {
                board.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }

    Ok(board)
}

pub fn enumerate_answers_slitherlink(
    url: &str,
    num_max_answers: usize,
) -> Result<(Board, Vec<Board>), &'static str> {
    let problem = slitherlink::deserialize_problem(url).ok_or("invalid url")?;
    let answer_common = slitherlink::solve_slitherlink(&problem).ok_or("no answer")?;
    let answers = slitherlink::enumerate_answers_slitherlink(&problem, num_max_answers);

    let height = problem.len();
    let width = problem[0].len();
    let mut board_common = Board::new(BoardKind::DotGrid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                board_common.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            }
        }
    }
    for y in 0..height {
        for x in 0..=width {
            if let Some(b) = answer_common.vertical[y][x] {
                board_common.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "black",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }
    for y in 0..=height {
        for x in 0..width {
            if let Some(b) = answer_common.horizontal[y][x] {
                board_common.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }

    let mut board_answers = vec![];
    for ans in answers {
        let mut board_answer = Board::new(BoardKind::Empty, height, width);
        // update board_answer according to ans
        for y in 0..height {
            for x in 0..=width {
                if answer_common.vertical[y][x].is_some() {
                    continue;
                }
                let b = ans.vertical[y][x];
                board_answer.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                });
            }
        }
        for y in 0..=height {
            for x in 0..width {
                if answer_common.horizontal[y][x].is_some() {
                    continue;
                }
                let b = ans.horizontal[y][x];
                board_answer.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                });
            }
        }

        board_answers.push(board_answer);
    }

    Ok((board_common, board_answers))
}
