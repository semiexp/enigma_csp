use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::{is_unique, Uniqueness};
use cspuz_rs::graph;
use cspuz_rs::puzzle::{ayeheya, heyawake};

pub fn solve_heyawake(url: &str, is_ayeheya: bool) -> Result<Board, &'static str> {
    let borders;
    let clues;
    let is_black;

    if is_ayeheya {
        let problem = ayeheya::deserialize_problem(url).ok_or("invalid url")?;
        borders = problem.0;
        clues = problem.1;
        if !ayeheya::all_room_symmetry(&borders) {
            return Err("asymmetry room");
        }
        is_black = ayeheya::solve_ayeheya(&borders, &clues).ok_or("no answer")?;
    } else {
        let problem = heyawake::deserialize_problem(url).ok_or("invalid url")?;
        borders = problem.0;
        clues = problem.1;
        is_black = heyawake::solve_heyawake(&borders, &clues).ok_or("no answer")?;
    }

    let height = is_black.len();
    let width = is_black[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_black));

    board.add_borders(&borders, "black");

    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black[y][x] {
                board.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }
    let rooms = graph::borders_to_rooms(&borders);
    assert_eq!(rooms.len(), clues.len());
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let (y, x) = rooms[i][0];
            board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
        }
    }

    Ok(board)
}

pub fn enumerate_answers_heyawake(
    url: &str,
    num_max_answers: usize,
) -> Result<(Board, Vec<Board>), &'static str> {
    let (borders, clues) = heyawake::deserialize_problem(url).ok_or("invalid url")?;
    let is_black_common = heyawake::solve_heyawake(&borders, &clues).ok_or("no answer")?;
    let answers = heyawake::enumerate_answers_heyawake(&borders, &clues, num_max_answers);

    let height = is_black_common.len();
    let width = is_black_common[0].len();

    let mut board_common = Board::new(BoardKind::Grid, height, width, Uniqueness::NotApplicable);
    board_common.add_borders(&borders, "black");
    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black_common[y][x] {
                board_common.push(Item::cell(
                    y,
                    x,
                    "#339933",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }
    let rooms = graph::borders_to_rooms(&borders);
    assert_eq!(rooms.len(), clues.len());
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let (y, x) = rooms[i][0];
            board_common.push(Item::cell(y, x, "black", ItemKind::Num(n)));
        }
    }

    let mut board_answers = vec![];
    for ans in answers {
        let mut board_answer =
            Board::new(BoardKind::Empty, height, width, Uniqueness::NotApplicable);
        for y in 0..height {
            for x in 0..width {
                if is_black_common[y][x].is_none() {
                    board_answer.push(Item::cell(
                        y,
                        x,
                        "#cccccc",
                        if ans[y][x] {
                            ItemKind::Block
                        } else {
                            ItemKind::Dot
                        },
                    ));
                }
            }
        }
        board_answers.push(board_answer);
    }

    Ok((board_common, board_answers))
}
