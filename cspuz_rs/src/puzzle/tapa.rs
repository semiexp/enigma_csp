use super::util;
use crate::graph;
use crate::serializer::{
    from_base36, problem_to_url, to_base36, url_to_problem, Choice, Combinator, Context, Grid,
    Optionalize, Spaces,
};
use crate::solver::{all, any, count_true, Solver, FALSE};

const EIGHT_NEIGHBORS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
];

pub fn solve_tapa(clues: &[Vec<Option<[i32; 4]>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    graph::active_vertices_connected_2d(&mut solver, is_black);

    solver.add_expr(!is_black.conv2d_and((2, 2)));

    for y in 0..h {
        for x in 0..w {
            if let Some(clue) = clues[y][x] {
                solver.add_expr(!is_black.at((y, x)));

                let mut neighbors = vec![];
                for &(dy, dx) in &EIGHT_NEIGHBORS {
                    let y2 = y as i32 + dy;
                    let x2 = x as i32 + dx;
                    if 0 <= y2 && y2 < h as i32 && 0 <= x2 && x2 < w as i32 {
                        neighbors.push(is_black.at((y2 as usize, x2 as usize)).expr());
                    } else {
                        neighbors.push(FALSE);
                    }
                }

                if clue[0] == -1 || clue[0] == 0 {
                    solver.add_expr(!any(&neighbors));
                    continue;
                }
                if clue[0] == 8 {
                    solver.add_expr(all(&neighbors));
                    continue;
                }

                let mut clue_counts = [0; 9];
                let mut total_clue_counts = 0;
                let mut has_any = false;
                for i in 0..4 {
                    if clue[i] != -1 {
                        assert!(clue[i] == -2 || 0 <= clue[i] && clue[i] <= 7);
                        if clue[i] == -2 {
                            has_any = true;
                        } else {
                            clue_counts[clue[i] as usize] += 1;
                        }
                        total_clue_counts += 1;
                    }
                }

                for l in 1..=8 {
                    if clue_counts[l] == 0 {
                        continue;
                    }
                    let mut conds = vec![];
                    for s in 0..8 {
                        let mut cond = vec![
                            !(neighbors[s].clone()),
                            !(neighbors[(s + l + 1) % 8].clone()),
                        ];
                        for i in 0..l {
                            cond.push(neighbors[(s + i + 1) % 8].clone());
                        }
                        conds.push(all(cond));
                    }
                    if has_any {
                        solver.add_expr(count_true(conds).ge(clue_counts[l]));
                    } else {
                        solver.add_expr(count_true(conds).eq(clue_counts[l]));
                    }
                }

                let mut unit_count = vec![];
                for s in 0..8 {
                    unit_count.push(&neighbors[s] & !&neighbors[(s + 1) % 8]);
                }
                solver.add_expr(count_true(unit_count).eq(total_clue_counts));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

struct TapaClueCombinator;

impl Combinator<[i32; 4]> for TapaClueCombinator {
    fn serialize(&self, _: &Context, input: &[[i32; 4]]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let clue = input[0];
        if clue[0] == 0 {
            return Some((1, vec!['0' as u8]));
        }
        let clue = clue.map(|n| if n == -2 { 0 } else { n });
        let mut num_clue = 0;
        for i in 0..4 {
            if clue[i] == -1 {
                break;
            }
            num_clue += 1;
        }
        let encoded = match num_clue {
            0 => return None,
            1 => vec![if clue[0] == 0 {
                '.' as u8
            } else {
                to_base36(clue[0])
            }],
            2 => {
                let v = clue[0] * 6 + clue[1] + 360;
                vec![to_base36(v / 36), to_base36(v % 36)]
            }
            3 => {
                let v = clue[0] * 16 + clue[1] * 4 + clue[2] + 396;
                vec![to_base36(v / 36), to_base36(v % 36)]
            }
            4 => {
                if clue[0] == 1 && clue[1] == 1 && clue[2] == 1 && clue[3] == 1 {
                    vec!['9' as u8]
                } else {
                    let v = clue[0] * 8 + clue[1] * 4 + clue[2] * 2 + clue[3] + 460;
                    vec![to_base36(v / 36), to_base36(v % 36)]
                }
            }
            _ => unreachable!(),
        };

        Some((1, encoded))
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<[i32; 4]>)> {
        if input.len() == 0 {
            return None;
        }
        if input[0] == '.' as u8 {
            return Some((1, vec![[-2, -1, -1, -1]]));
        }
        let c0 = from_base36(input[0])?;
        if 0 <= c0 && c0 <= 8 {
            return Some((1, vec![[c0, -1, -1, -1]]));
        } else if c0 == 9 {
            return Some((1, vec![[1, 1, 1, 1]]));
        } else {
            if input.len() < 2 {
                return None;
            }
            let v = c0 * 36 + from_base36(input[1])?;
            let decoded = if 360 <= v && v < 396 {
                [(v - 360) / 6, (v - 360) % 6, -1, -1]
            } else if 396 <= v && v < 460 {
                [(v - 396) / 16, (v - 396) / 4 % 4, (v - 396) % 4, -1]
            } else if 460 <= v && v < 476 {
                [
                    (v - 460) / 8,
                    (v - 460) / 4 % 2,
                    (v - 460) / 2 % 2,
                    (v - 460) % 2,
                ]
            } else {
                return None;
            };
            let decoded = decoded.map(|n| if n == 0 { -2 } else { n });
            Some((2, vec![decoded]))
        }
    }
}

pub type Problem = Vec<Vec<Option<[i32; 4]>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(TapaClueCombinator)),
        Box::new(Spaces::new(None, 'g')),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "tapa", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["tapa"], url)
}

#[cfg(test)]
mod tests {
    pub use super::*;

    fn problem_for_tests1() -> Problem {
        let height = 6;
        let width = 7;
        let mut ret: Problem = vec![vec![None; width]; height];

        ret[0][0] = Some([2, -1, -1, -1]);
        ret[1][2] = Some([1, 5, -1, -1]);
        ret[1][4] = Some([1, 1, 1, 1]);
        ret[4][1] = Some([8, -1, -1, -1]);
        ret[5][4] = Some([0, -1, -1, -1]);

        ret
    }

    fn problem_for_tests2() -> Problem {
        let height = 8;
        let width = 6;
        let mut ret: Problem = vec![vec![None; width]; height];

        ret[1][5] = Some([2, -1, -1, -1]);
        ret[2][1] = Some([1, 1, 1, 1]);
        ret[2][3] = Some([-2, -1, -1, -1]);
        ret[4][3] = Some([-2, -2, -2, -1]);
        ret[6][2] = Some([-2, -2, -1, -1]);
        ret[6][3] = Some([3, -2, -2, -1]);

        ret
    }

    #[test]
    fn test_tapa_problem1() {
        let problem = problem_for_tests1();
        let ans = solve_tapa(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = [
            [0, 1, 1, 1, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ];
        for y in 0..6 {
            for x in 0..7 {
                assert_eq!(
                    ans[y][x],
                    Some(expected[y][x] == 1),
                    "mismatch at ({}, {})",
                    y,
                    x
                );
            }
        }
    }

    #[test]
    fn test_tapa_problem2() {
        let problem = problem_for_tests2();
        let ans = solve_tapa(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_tapa_clue_combinator() {
        let ctx = &Context::new();
        let combinator = TapaClueCombinator;

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[[0, -1, -1, -1]]),
            Some((1, Vec::from("0")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[-2, -1, -1, -1]]),
            Some((1, Vec::from(".")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[3, -1, -1, -1]]),
            Some((1, Vec::from("3")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[1, 1, 1, 1]]),
            Some((1, Vec::from("9")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[1, 2, -1, -1]]),
            Some((1, Vec::from("a8")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[3, 3, -1, -1]]),
            Some((1, Vec::from("al")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[4, -2, -1, -1]]),
            Some((1, Vec::from("ao")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[2, -2, -2, -1]]),
            Some((1, Vec::from("bw")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[[1, 1, -2, -2]]),
            Some((1, Vec::from("d4")))
        );

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "0".as_bytes()),
            Some((1, vec![[0, -1, -1, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, ".0".as_bytes()),
            Some((1, vec![[-2, -1, -1, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "31".as_bytes()),
            Some((1, vec![[3, -1, -1, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "91".as_bytes()),
            Some((1, vec![[1, 1, 1, 1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "a8".as_bytes()),
            Some((2, vec![[1, 2, -1, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "al".as_bytes()),
            Some((2, vec![[3, 3, -1, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "ao".as_bytes()),
            Some((2, vec![[4, -2, -1, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "bw".as_bytes()),
            Some((2, vec![[2, -2, -2, -1]]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "d4".as_bytes()),
            Some((2, vec![[1, 1, -2, -2]]))
        );
    }

    #[test]
    fn test_nurimisaki_serializer() {
        let problem = problem_for_tests1();
        let url = "https://puzz.link/p?tapa/7/6/2nabg9w8o0h";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);

        let problem = problem_for_tests2();
        let url = "https://puzz.link/p?tapa/6/8/q2g9g.qb0pa0ccn";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
