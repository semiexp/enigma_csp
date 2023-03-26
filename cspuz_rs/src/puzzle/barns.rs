use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Combinator, Context, ContextBasedGrid, Map,
    MultiDigit, Rooms, Size, Tuple2,
};
use crate::solver::Solver;

pub fn solve_barns(
    icebarn: &[Vec<bool>],
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(icebarn);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let (is_passed, is_cross) = graph::crossable_single_cycle_grid_edges(&mut solver, &is_line);
    solver.add_expr(&is_passed);
    for y in 0..h {
        for x in 0..w {
            if icebarn[y][x] {
                if x != 0 && x != w - 1 {
                    solver.add_expr(
                        is_line
                            .horizontal
                            .at((y, x - 1))
                            .iff(is_line.horizontal.at((y, x))),
                    );
                }
                if y != 0 && y != h - 1 {
                    solver.add_expr(
                        is_line
                            .vertical
                            .at((y - 1, x))
                            .iff(is_line.vertical.at((y, x))),
                    );
                }
            } else {
                solver.add_expr(!is_cross.at((y, x)));
            }
        }
    }
    for y in 0..h {
        for x in 0..(w - 1) {
            if borders.vertical[y][x] {
                solver.add_expr(!is_line.horizontal.at((y, x)));
            }
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            if borders.horizontal[y][x] {
                solver.add_expr(!is_line.vertical.at((y, x)));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = (Vec<Vec<bool>>, graph::InnerGridEdges<Vec<Vec<bool>>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        ContextBasedGrid::new(Map::new(
            MultiDigit::new(2, 5),
            |x| Some(if x { 1 } else { 0 }),
            |x| Some(x == 1),
        )),
        Rooms,
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem.0);
    problem_to_url_with_context(
        combinator(),
        "barns",
        problem.clone(),
        &Context::sized(h, w),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["barns"], url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::InnerGridEdges;

    fn problem_for_tests() -> Problem {
        (
            crate::puzzle::util::tests::to_bool_2d([
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]),
            InnerGridEdges {
                horizontal: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]),
                vertical: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]),
            },
        )
    }

    #[test]
    fn test_barns_problem() {
        let (icebarn, borders) = problem_for_tests();
        let ans = solve_barns(&icebarn, &borders);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::GridEdges {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 1, 1],
                [0, 0, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 0, 1],
                [1, 1, 0, 1, 1],
                [0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_barns_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?barns/5/6/0gce000g00000g00";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
