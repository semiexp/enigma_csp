use super::util;
use crate::graph;
use crate::solver::{all, IntVarArray2D, Solver};

pub fn solve_scrabble(
    board: &[Vec<Option<i32>>],
    words: &[Vec<i32>],
    num_chars: i32,
    all_shown: bool,
) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = util::infer_shape(board);

    let mut solver = Solver::new();
    let answer = &solver.int_var_2d((h, w), -1, num_chars - 1);
    solver.add_answer_key_int(answer);

    add_constraints(&mut solver, answer, board, words, num_chars, all_shown);

    solver.irrefutable_facts().map(|f| f.get(answer))
}

pub fn enumerate_answers_scrabble(
    board: &[Vec<Option<i32>>],
    words: &[Vec<i32>],
    num_chars: i32,
    all_shown: bool,
    num_max_answers: usize,
) -> Vec<Vec<Vec<i32>>> {
    let (h, w) = util::infer_shape(board);

    let mut solver = Solver::new();
    let answer = &solver.int_var_2d((h, w), -1, num_chars - 1);
    solver.add_answer_key_int(answer);

    add_constraints(&mut solver, answer, board, words, num_chars, all_shown);

    solver
        .answer_iter()
        .take(num_max_answers)
        .map(|f| f.get_unwrap(answer))
        .collect()
}

fn add_constraints(
    solver: &mut Solver,
    answer: &IntVarArray2D,
    board: &[Vec<Option<i32>>],
    words: &[Vec<i32>],
    num_chars: i32,
    all_shown: bool,
) {
    let (h, w) = util::infer_shape(board);

    let is_passed = &solver.bool_var_2d((h, w));
    solver.add_expr(is_passed.iff(answer.ne(-1)));
    graph::active_vertices_connected_2d(solver, is_passed);

    let mut shown_chars = vec![false; num_chars as usize];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = board[y][x] {
                if n >= 0 {
                    shown_chars[n as usize] = true;
                }
            }
        }
    }

    let word_right = &solver.int_var_2d((h, w - 1), -1, words.len() as i32 - 1);
    let word_down = &solver.int_var_2d((h - 1, w), -1, words.len() as i32 - 1);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = board[y][x] {
                solver.add_expr(answer.at((y, x)).eq(n));
            } else {
                if all_shown {
                    for i in 0..num_chars {
                        if shown_chars[i as usize] {
                            solver.add_expr(answer.at((y, x)).ne(i));
                        }
                    }
                }
            }
            if x < w - 1 {
                if x > 0 {
                    solver.add_expr(word_right.at((y, x)).ne(-1).iff(
                        !is_passed.at((y, x - 1)) & is_passed.at((y, x)) & is_passed.at((y, x + 1)),
                    ));
                } else {
                    solver.add_expr(
                        word_right
                            .at((y, x))
                            .ne(-1)
                            .iff(is_passed.at((y, x)) & is_passed.at((y, x + 1))),
                    );
                }
                for i in 0..words.len() {
                    let word = &words[i];
                    if x + word.len() <= w {
                        let mut cand = vec![];
                        for j in 0..word.len() {
                            cand.push(answer.at((y, x + j)).eq(word[j]));
                        }
                        if x > 0 {
                            cand.push(answer.at((y, x - 1)).eq(-1));
                        }
                        if x + word.len() < w {
                            cand.push(answer.at((y, x + word.len())).eq(-1));
                        }
                        solver.add_expr(word_right.at((y, x)).eq(i as i32).iff(all(cand)));
                    } else {
                        solver.add_expr(word_right.at((y, x)).ne(i as i32));
                    }
                }
            }
            if y < h - 1 {
                if y > 0 {
                    solver.add_expr(word_down.at((y, x)).ne(-1).iff(
                        !is_passed.at((y - 1, x)) & is_passed.at((y, x)) & is_passed.at((y + 1, x)),
                    ));
                } else {
                    solver.add_expr(
                        word_down
                            .at((y, x))
                            .ne(-1)
                            .iff(is_passed.at((y, x)) & is_passed.at((y + 1, x))),
                    );
                }
                for i in 0..words.len() {
                    let word = &words[i];
                    if y + word.len() <= h {
                        let mut cand = vec![];
                        for j in 0..word.len() {
                            cand.push(answer.at((y + j, x)).eq(word[j]));
                        }
                        if y > 0 {
                            cand.push(answer.at((y - 1, x)).eq(-1));
                        }
                        if y + word.len() < h {
                            cand.push(answer.at((y + word.len(), x)).eq(-1));
                        }
                        solver.add_expr(word_down.at((y, x)).eq(i as i32).iff(all(cand)));
                    } else {
                        solver.add_expr(word_down.at((y, x)).ne(i as i32));
                    }
                }
            }
        }
    }
    for i in 0..(words.len() as i32) {
        solver.add_expr((word_right.eq(i).count_true() + word_down.eq(i).count_true()).eq(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scrabble_problem() {
        let board_orig = [".....", ".....", ".....", "....S"];
        let words_orig = ["HELLO", "WORLD", "ODDS"];

        let height = board_orig.len();
        let width = board_orig[0].len();
        let mut board = vec![vec![None; width]; height];
        for y in 0..height {
            let chars = board_orig[y].chars().collect::<Vec<_>>();
            for x in 0..width {
                if chars[x] != '.' {
                    board[y][x] = Some(chars[x] as i32 - 'A' as i32);
                }
            }
        }
        let mut words = vec![];
        for word in words_orig {
            words.push(
                word.chars()
                    .map(|c| c as i32 - 'A' as i32)
                    .collect::<Vec<_>>(),
            );
        }

        let ans = solve_scrabble(&board, &words, 26, true);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = ["HELLO", "....D", "WORLD", "....S"];
        for y in 0..height {
            let chars = expected[y].chars().collect::<Vec<_>>();
            for x in 0..width {
                let e = if chars[x] == '.' {
                    -1
                } else {
                    chars[x] as i32 - 'A' as i32
                };
                assert_eq!(ans[y][x], Some(e));
            }
        }
    }

    #[test]
    fn test_scrabble_problem_prefix() {
        let board = vec![vec![None; 3]; 2];
        let words = vec![vec![0, 0, 1], vec![0, 1]];
        let ans = solve_scrabble(&board, &words, 2, false);

        assert!(ans.is_some());
    }
}
