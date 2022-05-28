use crate::board::{Board, BoardKind, Item, ItemKind};
use cspuz_rs::puzzle::kropki::{self, KropkiClue};

pub fn solve_kropki(url: &str) -> Result<Board, &'static str> {
    let problem = kropki::deserialize_problem(url).ok_or("invalid url")?;
    let ans = kropki::solve_kropki(&problem).ok_or("no answer")?;

    let height = ans.len();
    let width = ans[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width);

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = ans[y][x] {
                board.push(Item::cell(y, x, "green", ItemKind::Num(n)));
            }
            if y < height - 1 {
                if problem.horizontal[y][x] == KropkiClue::White {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::SmallCircle,
                    });
                } else if problem.horizontal[y][x] == KropkiClue::Black {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::SmallFilledCircle,
                    });
                }
            }
            if x < width - 1 {
                if problem.vertical[y][x] == KropkiClue::White {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "black",
                        kind: ItemKind::SmallCircle,
                    });
                } else if problem.vertical[y][x] == KropkiClue::Black {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "black",
                        kind: ItemKind::SmallFilledCircle,
                    });
                }
            }
        }
    }

    Ok(board)
}
