use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::kropki_pairs::{self, KropkiClue};

pub fn solve_kropki_pairs(url: &str) -> Result<Board, &'static str> {
    let (walls, cells) = kropki_pairs::deserialize_problem(url).ok_or("invalid url")?;
    let ans = kropki_pairs::solve_kropki_pairs(&walls, &cells).ok_or("no answer")?;

    let height = ans.len();
    let width = ans[0].len();
    let mut board = Board::new(BoardKind::Grid, height, width, is_unique(&ans));

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = cells[y][x] {
                if n > 0 {
                    board.push(Item::cell(y, x, "black", ItemKind::Num(n)));
                } else {
                    board.push(Item::cell(y, x, "black", ItemKind::Fill));
                }
            } else if let Some(n) = ans[y][x] {
                board.push(Item::cell(y, x, "green", ItemKind::Num(n)));
            }
            if y < height - 1 {
                if walls.horizontal[y][x] == KropkiClue::White {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::SmallCircle,
                    });
                } else if walls.horizontal[y][x] == KropkiClue::Black {
                    board.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "black",
                        kind: ItemKind::SmallFilledCircle,
                    });
                }
            }
            if x < width - 1 {
                if walls.vertical[y][x] == KropkiClue::White {
                    board.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "black",
                        kind: ItemKind::SmallCircle,
                    });
                } else if walls.vertical[y][x] == KropkiClue::Black {
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
