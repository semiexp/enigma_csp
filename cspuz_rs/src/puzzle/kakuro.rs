use super::util;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, ContextBasedGrid,
    Dict, Optionalize, Size, Spaces, Tuple2, UnlimitedSeq,
};
use crate::solver::{all, any, IntVarArray1D, Solver};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KakuroClue {
    pub down: Option<i32>,
    pub right: Option<i32>,
}

pub fn solve_kakuro(clues: &[Vec<Option<KakuroClue>>]) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let numbers = &solver.int_var_2d((h, w), 0, 9);
    solver.add_answer_key_int(numbers);

    for y in 0..h {
        for x in 0..w {
            if clues[y][x].is_some() {
                solver.add_expr(numbers.at((y, x)).eq(0));
            } else {
                solver.add_expr(numbers.at((y, x)).ne(0));
            }
        }
    }

    let mut dict = vec![vec![vec![]; 46]; 10];
    for b in 1i32..512 {
        let num = b.count_ones();
        let mut sum = 0;
        for i in 0..9 {
            if (b & (1 << i)) != 0 {
                sum += i + 1;
            }
        }
        dict[num as usize][sum as usize].push(b);
    }

    let mut add_constraints = |cells: IntVarArray1D, clue: Option<i32>| -> bool {
        solver.all_different(&cells);

        if let Some(n) = clue {
            let n_cells = cells.len() as i32;
            let n_min = n_cells * (n_cells + 1) / 2;
            let n_max = n_cells * 10 - n_min;
            if !(n_min <= n && n <= n_max) {
                return false;
            }

            let appear = solver.bool_var_1d(9);
            for i in 0..9 {
                solver.add_expr(appear.at(i).iff(cells.eq(i as i32 + 1).any()));
            }

            let mut cands = vec![];
            for b in &dict[cells.len()][n as usize] {
                let mut lits = vec![];
                for i in 0..9 {
                    if (b & (1 << i)) != 0 {
                        lits.push(appear.at(i).expr());
                    } else {
                        lits.push(!appear.at(i));
                    }
                }
                cands.push(all(lits));
            }
            solver.add_expr(any(cands));
        }

        true
    };

    for y in 0..h {
        for x in 0..w {
            if let Some(clue) = clues[y][x] {
                // down
                let mut y2 = y + 1;
                while y2 < h && clues[y2][x].is_none() {
                    y2 += 1;
                }
                if y2 - y >= 2 {
                    if !add_constraints(numbers.slice_fixed_x(((y + 1)..y2, x)), clue.down) {
                        return None;
                    }
                }

                // right
                let mut x2 = x + 1;
                while x2 < w && clues[y][x2].is_none() {
                    x2 += 1;
                }
                if x2 - x >= 2 {
                    if !add_constraints(numbers.slice_fixed_y((y, (x + 1)..x2)), clue.right) {
                        return None;
                    }
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(numbers))
}

struct KakuroNumCombinator;

impl Combinator<Option<i32>> for KakuroNumCombinator {
    fn serialize(&self, _: &Context, input: &[Option<i32>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let n = input[0];

        if n.is_none() {
            return Some((1, vec!['-' as u8]));
        }

        let n = n.unwrap();
        let c = if 1 <= n && n <= 9 {
            n as u8 + '0' as u8
        } else if 10 <= n && n <= 19 {
            n as u8 - 10 + 'a' as u8
        } else if 20 <= n && n <= 45 {
            n as u8 - 20 + 'A' as u8
        } else {
            return None;
        };

        Some((1, vec![c]))
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<Option<i32>>)> {
        if input.len() == 0 {
            return None;
        }
        let c = input[0];

        if c == '-' as u8 {
            return Some((1, vec![None]));
        }

        let v = if '0' as u8 <= c && c <= '9' as u8 {
            c - '0' as u8
        } else if 'a' as u8 <= c && c <= 'j' as u8 {
            c - 'a' as u8 + 10
        } else if 'A' as u8 <= c && c <= 'Z' as u8 {
            c - 'A' as u8 + 20
        } else {
            return None;
        };

        Some((1, vec![Some(v as i32)]))
    }
}

pub type Problem = Vec<Vec<Option<KakuroClue>>>;

type IntermediateProblem = (
    Vec<Vec<Option<(Option<i32>, Option<i32>)>>>,
    Vec<Option<i32>>,
);

fn combinator() -> impl Combinator<IntermediateProblem> {
    Size::new(Tuple2::new(
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Optionalize::new(Tuple2::new(
                KakuroNumCombinator,
                KakuroNumCombinator,
            ))),
            Box::new(Dict::new(Some((None, None)), ".")),
            Box::new(Spaces::new(None, 'k')),
        ])),
        UnlimitedSeq::new(KakuroNumCombinator),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(problem);
    if !(h >= 2 && w >= 2) {
        return None;
    }

    let mut intermediate_grid = vec![vec![None; w - 1]; h - 1];
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            intermediate_grid[y][x] = problem[y + 1][x + 1].map(|clue| (clue.down, clue.right));
        }
    }

    let mut rem_seq = vec![];
    for x in 1..w {
        if problem[1][x].is_none() {
            if problem[0][x].is_none() {
                return None;
            }
            rem_seq.push(problem[0][x].unwrap().down);
        }
    }
    for y in 1..h {
        if problem[y][1].is_none() {
            if problem[y][0].is_none() {
                return None;
            }
            rem_seq.push(problem[y][0].unwrap().right);
        }
    }

    problem_to_url_with_context(
        combinator(),
        "kakuro",
        (intermediate_grid, rem_seq),
        &Context::sized(h - 1, w - 1),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let (intermediate_grid, rem_seq) = url_to_problem(combinator(), &["kakuro"], url)?;

    let (h, w) = util::infer_shape(&intermediate_grid);
    let h = h + 1;
    let w = w + 1;

    let mut ret = vec![vec![None; w]; h];
    for y in 1..h {
        for x in 1..w {
            ret[y][x] =
                intermediate_grid[y - 1][x - 1].map(|(down, right)| KakuroClue { down, right });
        }
    }
    let mut idx = 0;
    ret[0][0] = Some(KakuroClue {
        down: None,
        right: None,
    });
    for x in 1..w {
        if ret[1][x].is_none() {
            if idx >= rem_seq.len() {
                return None;
            }
            ret[0][x] = Some(KakuroClue {
                down: rem_seq[idx],
                right: None,
            });
            idx += 1;
        } else {
            ret[0][x] = Some(KakuroClue {
                down: None,
                right: None,
            });
        }
    }
    for y in 1..h {
        if ret[y][1].is_none() {
            if idx >= rem_seq.len() {
                return None;
            }
            ret[y][0] = Some(KakuroClue {
                down: None,
                right: rem_seq[idx],
            });
            idx += 1;
        } else {
            ret[y][0] = Some(KakuroClue {
                down: None,
                right: None,
            });
        }
    }

    Some(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Vec<Vec<Option<KakuroClue>>> {
        // https://puzz.link/p?kakuro/6/5/Dclh4t9fl3-p-gl-alJeC3BgG
        let mut ret = vec![vec![None; 7]; 6];
        ret[0][0] = Some(KakuroClue {
            down: None,
            right: None,
        });
        ret[0][1] = Some(KakuroClue {
            down: None,
            right: None,
        });
        ret[0][2] = Some(KakuroClue {
            down: Some(29),
            right: None,
        });
        ret[0][3] = Some(KakuroClue {
            down: Some(14),
            right: None,
        });
        ret[0][4] = Some(KakuroClue {
            down: None,
            right: None,
        });
        ret[0][5] = Some(KakuroClue {
            down: Some(22),
            right: None,
        });
        ret[0][6] = Some(KakuroClue {
            down: Some(3),
            right: None,
        });
        ret[1][0] = Some(KakuroClue {
            down: None,
            right: None,
        });
        ret[1][1] = Some(KakuroClue {
            down: Some(23),
            right: Some(12),
        });
        ret[1][4] = Some(KakuroClue {
            down: Some(17),
            right: Some(4),
        });
        ret[2][0] = Some(KakuroClue {
            down: None,
            right: Some(21),
        });
        ret[3][0] = Some(KakuroClue {
            down: None,
            right: Some(16),
        });
        ret[3][3] = Some(KakuroClue {
            down: Some(9),
            right: Some(15),
        });
        ret[3][6] = Some(KakuroClue {
            down: Some(3),
            right: None,
        });
        ret[4][0] = Some(KakuroClue {
            down: None,
            right: Some(26),
        });
        ret[5][0] = Some(KakuroClue {
            down: None,
            right: None,
        });
        ret[5][1] = Some(KakuroClue {
            down: None,
            right: Some(16),
        });
        ret[5][4] = Some(KakuroClue {
            down: None,
            right: Some(10),
        });

        ret
    }

    #[test]
    fn test_kakuro_problem() {
        let problem = problem_for_tests();
        let ans = solve_kakuro(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_2d([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 9, 0, 3, 1],
            [0, 6, 4, 5, 3, 1, 2],
            [0, 9, 7, 0, 9, 6, 0],
            [0, 8, 6, 2, 5, 4, 1],
            [0, 0, 9, 7, 0, 8, 2],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_kakuro_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?kakuro/6/5/Dclh4t9fl3-p-gl-alJeC3BgG";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
