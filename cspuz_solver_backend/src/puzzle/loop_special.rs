use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::loop_special::{self, LoopSpecialClue};

pub fn solve_loop_speical(url: &str) -> Result<Board, &'static str> {
    let problem = loop_special::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = loop_special::solve_loop_special(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&is_line));

    board.add_lines_irrefutable_facts(&is_line, "green", None);

    for y in 0..height {
        for x in 0..width {
            match problem[y][x] {
                LoopSpecialClue::Num(n) => {
                    board.push(Item::cell(y, x, "black", ItemKind::Circle));
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                }
                LoopSpecialClue::Empty => (),
                _ => {
                    let (up, down, left, right) = match problem[y][x] {
                        LoopSpecialClue::Cross => (true, true, true, true),
                        LoopSpecialClue::Vertical => (true, true, false, false),
                        LoopSpecialClue::Horizontal => (false, false, true, true),
                        LoopSpecialClue::UpRight => (true, false, false, true),
                        LoopSpecialClue::UpLeft => (true, false, true, false),
                        LoopSpecialClue::DownLeft => (false, true, true, false),
                        LoopSpecialClue::DownRight => (false, true, false, true),
                        _ => unreachable!(),
                    };
                    if up {
                        board.push(Item {
                            y: y * 2,
                            x: x * 2 + 1,
                            color: "black",
                            kind: ItemKind::Line,
                        });
                    }
                    if down {
                        board.push(Item {
                            y: y * 2 + 2,
                            x: x * 2 + 1,
                            color: "black",
                            kind: ItemKind::Line,
                        });
                    }
                    if left {
                        board.push(Item {
                            y: y * 2 + 1,
                            x: x * 2,
                            color: "black",
                            kind: ItemKind::Line,
                        });
                    }
                    if right {
                        board.push(Item {
                            y: y * 2 + 1,
                            x: x * 2 + 2,
                            color: "black",
                            kind: ItemKind::Line,
                        });
                    }
                }
            }
        }
    }

    Ok(board)
}
