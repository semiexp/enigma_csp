use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, Choice,
    Combinator, Dict, KudamonoGrid,
};
use crate::solver::{any, Solver};

pub fn solve_milktea(clues: &[Vec<i32>]) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_center = &solver.bool_var_2d((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut adj = vec![];

            if y > 0 {
                adj.push(is_line.vertical.at((y - 1, x)));
            }
            if x > 0 {
                adj.push(is_line.horizontal.at((y, x - 1)));
            }
            if y < h - 1 {
                adj.push(is_line.vertical.at((y, x)));
            }
            if x < w - 1 {
                adj.push(is_line.horizontal.at((y, x)));
            }

            let n = &solver.int_var(0, 3);
            solver.add_expr(is_line.vertex_neighbors((y, x)).count_true().eq(n));

            if clues[y][x] > 0 {
                solver.add_expr(!is_center.at((y, x)));
                solver.add_expr(n.eq(1));
            } else {
                solver.add_expr(is_center.at((y, x)).imp(n.eq(3)));
                solver.add_expr((!is_center.at((y, x))).imp(n.eq(0) | n.eq(2)));
            }
        }
    }

    let ty_horizontal = &solver.int_var_2d((h, w - 1), 0, 4);
    let ty_vertical = &solver.int_var_2d((h - 1, w), 0, 4);
    solver.add_expr(is_line.horizontal.iff(ty_horizontal.ne(0)));
    solver.add_expr(is_line.vertical.iff(ty_vertical.ne(0)));

    for y in 0..h {
        for x in 0..(w - 1) {
            for i in 0..2 {
                solver.add_expr(ty_horizontal.at((y, x)).eq(i * 2 + 1).imp(
                    ty_horizontal.at_offset((y, x), (0, -1), 0).eq(i * 2 + 1)
                        ^ (clues[y][x] == i + 1),
                ));
                solver.add_expr(ty_horizontal.at((y, x)).eq(i * 2 + 1).imp(
                    ty_horizontal.at_offset((y, x), (0, 1), 0).eq(i * 2 + 1)
                        ^ is_center.at((y, x + 1)),
                ));

                solver.add_expr(ty_horizontal.at((y, x)).eq(i * 2 + 2).imp(
                    ty_horizontal.at_offset((y, x), (0, -1), 0).eq(i * 2 + 2)
                        ^ is_center.at((y, x)),
                ));
                solver.add_expr(ty_horizontal.at((y, x)).eq(i * 2 + 2).imp(
                    ty_horizontal.at_offset((y, x), (0, 1), 0).eq(i * 2 + 2)
                        ^ (clues[y][x + 1] == i + 1),
                ));
            }
        }
    }

    for y in 0..(h - 1) {
        for x in 0..w {
            for i in 0..2 {
                solver.add_expr(ty_vertical.at((y, x)).eq(i * 2 + 1).imp(
                    ty_vertical.at_offset((y, x), (-1, 0), 0).eq(i * 2 + 1)
                        ^ (clues[y][x] == i + 1),
                ));
                solver.add_expr(ty_vertical.at((y, x)).eq(i * 2 + 1).imp(
                    ty_vertical.at_offset((y, x), (1, 0), 0).eq(i * 2 + 1)
                        ^ is_center.at((y + 1, x)),
                ));

                solver.add_expr(ty_vertical.at((y, x)).eq(i * 2 + 2).imp(
                    ty_vertical.at_offset((y, x), (-1, 0), 0).eq(i * 2 + 2) ^ is_center.at((y, x)),
                ));
                solver.add_expr(ty_vertical.at((y, x)).eq(i * 2 + 2).imp(
                    ty_vertical.at_offset((y, x), (1, 0), 0).eq(i * 2 + 2)
                        ^ (clues[y + 1][x] == i + 1),
                ));
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            let mut pat = vec![];

            if 0 < y && y < h - 1 {
                if x > 0 {
                    pat.push(
                        ty_vertical.at((y - 1, x)).eq(1)
                            & ty_vertical.at((y, x)).eq(2)
                            & ty_horizontal.at((y, x - 1)).eq(3),
                    );
                    pat.push(
                        ty_vertical.at((y - 1, x)).eq(3)
                            & ty_vertical.at((y, x)).eq(4)
                            & ty_horizontal.at((y, x - 1)).eq(1),
                    );
                }
                if x < w - 1 {
                    pat.push(
                        ty_vertical.at((y - 1, x)).eq(1)
                            & ty_vertical.at((y, x)).eq(2)
                            & ty_horizontal.at((y, x)).eq(4),
                    );
                    pat.push(
                        ty_vertical.at((y - 1, x)).eq(3)
                            & ty_vertical.at((y, x)).eq(4)
                            & ty_horizontal.at((y, x)).eq(2),
                    );
                }
            }
            if 0 < x && x < w - 1 {
                if y > 0 {
                    pat.push(
                        ty_horizontal.at((y, x - 1)).eq(1)
                            & ty_horizontal.at((y, x)).eq(2)
                            & ty_vertical.at((y - 1, x)).eq(3),
                    );
                    pat.push(
                        ty_horizontal.at((y, x - 1)).eq(3)
                            & ty_horizontal.at((y, x)).eq(4)
                            & ty_vertical.at((y - 1, x)).eq(1),
                    );
                }
                if y < h - 1 {
                    pat.push(
                        ty_horizontal.at((y, x - 1)).eq(1)
                            & ty_horizontal.at((y, x)).eq(2)
                            & ty_vertical.at((y, x)).eq(4),
                    );
                    pat.push(
                        ty_horizontal.at((y, x - 1)).eq(3)
                            & ty_horizontal.at((y, x)).eq(4)
                            & ty_vertical.at((y, x)).eq(2),
                    );
                }
            }

            solver.add_expr(is_center.at((y, x)).imp(any(pat)));
        }
    }
    solver.irrefutable_facts().map(|f| f.get(is_line))
}

pub type Problem = Vec<Vec<i32>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(
        Choice::new(vec![
            Box::new(Dict::new(1, "w")),
            Box::new(Dict::new(2, "b")),
        ]),
        0,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "milk-tea", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let info = get_kudamono_url_info(url)?;
    kudamono_url_info_to_problem(combinator(), info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![1, 0, 0, 0, 1, 0],
            vec![0, 2, 0, 2, 0, 2],
            vec![0, 1, 0, 2, 0, 0],
            vec![1, 0, 0, 0, 1, 0],
            vec![2, 0, 2, 2, 0, 0],
        ]
    }

    #[test]
    fn test_milktea_problem() {
        let problem = problem_for_tests();
        let ans = solve_milktea(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_milktea_exhaustive() {
        for pat in 1..81 {
            let mut clues = vec![vec![0; 3]; 3];
            clues[0][1] = pat / 27;
            clues[1][0] = (pat / 9) % 3;
            clues[1][2] = (pat / 3) % 3;
            clues[2][1] = pat % 3;

            let n_ones = clues
                .iter()
                .map(|row| row.iter().filter(|&&x| x == 1).count())
                .sum::<usize>();
            let n_twos = clues
                .iter()
                .map(|row| row.iter().filter(|&&x| x == 2).count())
                .sum::<usize>();

            let expected = ((n_ones == 1 && n_twos == 2) || (n_ones == 2 && n_twos == 1))
                && (clues[0][1] == clues[2][1] || clues[1][0] == clues[1][2]);
            let actual = solve_milktea(&clues).is_some();

            assert_eq!(actual, expected, "{:?}", clues);
        }
    }

    #[test]
    fn test_milktea_serializer() {
        let problem = problem_for_tests();
        let url =
            "https://pedros.works/paper-puzzle-player?W=6x5&L=b0w1w3w3b1b2b5b2b1w3w3b4&G=milk-tea";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
