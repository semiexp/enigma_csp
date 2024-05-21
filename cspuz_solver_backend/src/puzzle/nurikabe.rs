use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::{is_unique, Uniqueness};
use cspuz_rs::puzzle::nurikabe;

pub fn solve_nurikabe(url: &str) -> Result<Board, &'static str> {
    let problem = nurikabe::deserialize_problem(url).ok_or("invalid url")?;
    let ans = nurikabe::solve_nurikabe(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&ans));
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                if clue > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::Text("?")));
                }
            } else if let Some(a) = ans[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(board)
}

pub fn enumerate_answers_nurikabe(
    url: &str,
    num_max_answers: usize,
) -> Result<(Board, Vec<Board>), &'static str> {
    let problem = nurikabe::deserialize_problem(url).ok_or("invalid url")?;
    let ans_common = nurikabe::solve_nurikabe(&problem).ok_or("no answer")?;
    let answers = nurikabe::enumerate_answers_nurikabe(&problem, num_max_answers);

    let height = problem.len();
    let width = problem[0].len();
    let mut board_common = Board::new(BoardKind::Grid, height, width, Uniqueness::NotApplicable);

    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                if clue > 0 {
                    board_common.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                } else {
                    board_common.push(Item::cell(y, x, "black", ItemKind::Text("?")));
                }
            } else if let Some(a) = ans_common[y][x] {
                board_common.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    let mut boards = vec![];
    for ans in answers {
        let mut board_answer =
            Board::new(BoardKind::Empty, height, width, Uniqueness::NotApplicable);
        for y in 0..height {
            for x in 0..width {
                if ans_common[y][x].is_some() {
                    continue;
                }
                if let Some(clue) = problem[y][x] {
                    if clue > 0 {
                        board_answer.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                    } else {
                        board_answer.push(Item::cell(y, x, "black", ItemKind::Text("?")));
                    }
                } else {
                    let a = ans[y][x];
                    board_answer.push(Item::cell(
                        y,
                        x,
                        "green",
                        if a { ItemKind::Block } else { ItemKind::Dot },
                    ));
                }
            }
        }
        boards.push(board_answer);
    }

    Ok((board_common, boards))
}
