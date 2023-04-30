use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, ContextBasedGrid,
    HexInt, Map, MultiDigit, Optionalize, Size, Spaces, Tuple2,
};
use crate::solver::{count_true, BoolExpr, Solver};

pub fn solve_icewalk(
    icebarn: &[Vec<bool>],
    num: &[Vec<Option<i32>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(icebarn);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let (is_passed, is_cross) = graph::crossable_single_cycle_grid_edges(&mut solver, &is_line);
    for y in 0..h {
        for x in 0..w {
            if num[y][x].is_some() {
                solver.add_expr(is_passed.at((y, x)));
            }
            if icebarn[y][x] {
                if x == 0 {
                    solver.add_expr(!is_line.horizontal.at((y, x)));
                } else if x == w - 1 {
                    solver.add_expr(!is_line.horizontal.at((y, x - 1)));
                } else {
                    solver.add_expr(
                        is_line
                            .horizontal
                            .at((y, x - 1))
                            .iff(is_line.horizontal.at((y, x))),
                    );
                }
                if y == 0 {
                    solver.add_expr(!is_line.vertical.at((y, x)));
                } else if y == h - 1 {
                    solver.add_expr(!is_line.vertical.at((y - 1, x)));
                } else {
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

    let direction = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    let up = &(&is_line.vertical & &direction.vertical);
    let down = &(&is_line.vertical & !&direction.vertical);
    let left = &(&is_line.horizontal & &direction.horizontal);
    let right = &(&is_line.horizontal & !&direction.horizontal);

    let line_size = &solver.int_var_2d((h, w), 0, (h * w) as i32);
    let line_rank = &solver.int_var_2d((h, w), 0, (h * w) as i32);

    let mut add_constraint = |src: (usize, usize), dest: (usize, usize), edge: BoolExpr| match (
        icebarn[src.0][src.1],
        icebarn[dest.0][dest.1],
    ) {
        (false, false) => {
            solver.add_expr(edge.imp(
                line_size.at(src).eq(line_size.at(dest))
                    & line_rank.at(src).eq(line_rank.at(dest) + 1),
            ));
        }
        (false, true) => {
            solver.add_expr(edge.imp(line_rank.at(src).eq(0)));
        }
        (true, false) => {
            solver.add_expr(edge.imp(line_rank.at(dest).eq(line_size.at(dest))));
        }
        (true, true) => (),
    };

    for y in 0..h {
        for x in 0..w {
            if y > 0 {
                add_constraint((y, x), (y - 1, x), up.at((y - 1, x)));
            }
            if y < h - 1 {
                add_constraint((y, x), (y + 1, x), down.at((y, x)));
            }
            if x > 0 {
                add_constraint((y, x), (y, x - 1), left.at((y, x - 1)));
            }
            if x < w - 1 {
                add_constraint((y, x), (y, x + 1), right.at((y, x)));
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = num[y][x] {
                solver.add_expr(line_size.at((y, x)).eq(n - 1));
            }

            let mut inbound = vec![];
            let mut outbound = vec![];
            if y > 0 {
                inbound.push(is_line.vertical.at((y - 1, x)) & !direction.vertical.at((y - 1, x)));
                outbound.push(up.at((y - 1, x)));
            }
            if y < h - 1 {
                inbound.push(is_line.vertical.at((y, x)) & direction.vertical.at((y, x)));
                outbound.push(down.at((y, x)));
            }
            if x > 0 {
                inbound
                    .push(is_line.horizontal.at((y, x - 1)) & !direction.horizontal.at((y, x - 1)));
                outbound.push(left.at((y, x - 1)));
            }
            if x < w - 1 {
                inbound.push(is_line.horizontal.at((y, x)) & direction.horizontal.at((y, x)));
                outbound.push(right.at((y, x)));
            }
            solver.add_expr(count_true(&inbound).eq(count_true(&outbound)));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = (Vec<Vec<bool>>, Vec<Vec<Option<i32>>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(Tuple2::new(
        ContextBasedGrid::new(Map::new(
            MultiDigit::new(2, 5),
            |x| Some(if x { 1 } else { 0 }),
            |x| Some(x == 1),
        )),
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Optionalize::new(HexInt)),
            Box::new(Spaces::new(None, 'g')),
        ])),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem.0);
    problem_to_url_with_context(
        combinator(),
        "icewalk",
        problem.clone(),
        &Context::sized(h, w),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["icewalk"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        (
            crate::puzzle::util::tests::to_bool_2d([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]),
            vec![
                vec![None, None, None, None, None, None],
                vec![Some(2), None, None, Some(2), None, None],
                vec![None, None, None, Some(3), None, None],
                vec![None, None, None, None, None, None],
                vec![None, None, Some(5), None, Some(1), None],
                vec![None, None, None, None, Some(3), None],
                vec![None, None, None, None, None, Some(3)],
            ],
        )
    }

    #[test]
    fn test_icewalk_problem() {
        let (icebarn, num) = problem_for_tests();
        let ans = solve_icewalk(&icebarn, &num);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::GridEdges {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [1, 1, 0, 0, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_icewalk_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?icewalk/6/7/g63845qg0l2h2k3p5g1k3l3";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
