use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, Combinator,
    Dict, KudamonoGrid,
};
use crate::solver::{all, count_true, Solver};

pub fn solve_tricklayer(
    is_block: &[Vec<bool>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(is_block);

    let mut solver = Solver::new();
    let edges = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&edges.horizontal);
    solver.add_answer_key_bool(&edges.vertical);

    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && (is_block[y][x] || is_block[y + 1][x]) {
                solver.add_expr(edges.horizontal.at((y, x)));
            }
            if x < w - 1 && (is_block[y][x] || is_block[y][x + 1]) {
                solver.add_expr(edges.vertical.at((y, x)));
            }
        }
    }

    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            let up = &edges.vertical.at((y, x));
            let down = &edges.vertical.at((y + 1, x));
            let left = &edges.horizontal.at((y, x));
            let right = &edges.horizontal.at((y, x + 1));
            solver.add_expr(!((up ^ down) & (left ^ right)));
            solver.add_expr(count_true([up, down, left, right]).ne(1));
        }
    }

    let to_right = solver.int_var_2d((h, w), 0, w as i32 - 1);
    let to_down = solver.int_var_2d((h, w), 0, h as i32 - 1);

    solver.add_expr(to_right.slice_fixed_x((.., w - 1)).eq(0));
    solver.add_expr(
        to_right
            .slice((.., ..(w - 1)))
            .eq(edges.vertical.ite(0, to_right.slice((.., 1..)) + 1)),
    );
    solver.add_expr(to_down.slice_fixed_y((h - 1, ..)).eq(0));
    solver.add_expr(
        to_down
            .slice((..(h - 1), ..))
            .eq(edges.horizontal.ite(0, to_down.slice((1.., ..)) + 1)),
    );
    for y1 in 0..h {
        for x1 in 0..w {
            if is_block[y1][x1] {
                continue;
            }
            for y2 in 0..h {
                for x2 in 0..w {
                    if is_block[y2][x2] {
                        continue;
                    }
                    if (y1, x1) < (y2, x2) {
                        let mut both_corner = vec![];
                        if y1 > 0 {
                            both_corner.push(edges.horizontal.at((y1 - 1, x1)));
                        }
                        if x1 > 0 {
                            both_corner.push(edges.vertical.at((y1, x1 - 1)));
                        }
                        if y2 > 0 {
                            both_corner.push(edges.horizontal.at((y2 - 1, x2)));
                        }
                        if x2 > 0 {
                            both_corner.push(edges.vertical.at((y2, x2 - 1)));
                        }
                        solver.add_expr(all(&both_corner).imp(
                            to_right.at((y1, x1)).ne(to_right.at((y2, x2)))
                                | to_down.at((y1, x1)).ne(to_down.at((y2, x2))),
                        ));
                        solver.add_expr(all(&both_corner).imp(
                            to_right.at((y1, x1)).ne(to_down.at((y2, x2)))
                                | to_down.at((y1, x1)).ne(to_right.at((y2, x2))),
                        ));
                    }
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(edges))
}

type Problem = Vec<Vec<bool>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(Dict::new(true, "x"), false)
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "tricklayer", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let info = get_kudamono_url_info(url)?;
    kudamono_url_info_to_problem(combinator(), info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        crate::puzzle::util::tests::to_bool_2d([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
    }

    #[test]
    fn test_tricklayer_problem() {
        let problem = problem_for_tests();

        let ans = solve_tricklayer(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_tricklayer_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=4&H=3&L=x1x2x8x7x1&G=tricklayer";
        crate::puzzle::util::tests::serializer_test(
            problem,
            url,
            serialize_problem,
            deserialize_problem,
        );
    }
}
