use crate::board::{Board, BoardKind, Item, ItemKind};
use crate::uniqueness::is_unique;
use cspuz_rs::puzzle::letter_weights;

const ALPHA: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub fn solve_letter_weights(url: &str) -> Result<Board, &'static str> {
    let (sums, chars, nums) = letter_weights::deserialize_problem(url)?;
    let (chars, nums, ans) = letter_weights::solve_letter_weights(&sums, &chars, &nums);
    let ans = ans.ok_or("no answer")?;

    let mut board = Board::new(
        BoardKind::Grid,
        chars.len() + 1,
        nums.len() + 1,
        is_unique(&ans),
    );

    for y in 0..chars.len() {
        assert!('A' <= chars[y] && chars[y] <= 'Z');
        let i = (chars[y] as u8 - 'A' as u8) as usize;
        board.push(Item::cell(y + 1, 0, "black", ItemKind::Text(&ALPHA[i..=i])));
    }

    for x in 0..nums.len() {
        board.push(Item::cell(0, x + 1, "black", ItemKind::Num(nums[x])));
    }

    for y in 0..chars.len() {
        for x in 0..nums.len() {
            if let Some(clue) = ans[y][x] {
                board.push(Item::cell(
                    y + 1,
                    x + 1,
                    "green",
                    if clue {
                        ItemKind::Circle
                    } else {
                        ItemKind::Dot
                    },
                ));
            }
        }
    }

    Ok(board)
}
