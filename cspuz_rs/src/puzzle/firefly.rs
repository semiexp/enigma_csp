use super::util;
use crate::graph;
use crate::items::{Arrow, NumberedArrow};
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Grid, MaybeSkip, NumberedArrowCombinator,
    Optionalize, Spaces,
};
use crate::solver::{count_true, BoolVar, IntVar, Solver};

pub fn solve_firefly(
    clues: &[Vec<Option<NumberedArrow>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let line_ul = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    let line_dr = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_expr(
        is_line
            .as_sequence()
            .iff(line_ul.as_sequence() | line_dr.as_sequence()),
    );
    solver.add_expr(!(line_ul.as_sequence() & line_dr.as_sequence()));

    // unicyclic
    let ignored_edge = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_expr(ignored_edge.as_sequence().count_true().eq(1));
    let rank = &solver.int_var_2d((h, w), 0, (h * w - 1) as i32);
    solver.add_expr(
        (&line_ul.horizontal & !&ignored_edge.horizontal)
            .imp(rank.slice((.., ..(w - 1))).lt(rank.slice((.., 1..)))),
    );
    solver.add_expr(
        (&line_ul.vertical & !&ignored_edge.vertical)
            .imp(rank.slice((..(h - 1), ..)).lt(rank.slice((1.., ..)))),
    );
    solver.add_expr(
        (&line_dr.horizontal & !&ignored_edge.horizontal)
            .imp(rank.slice((.., ..(w - 1))).gt(rank.slice((.., 1..)))),
    );
    solver.add_expr(
        (&line_dr.vertical & !&ignored_edge.vertical)
            .imp(rank.slice((..(h - 1), ..)).gt(rank.slice((1.., ..)))),
    );

    let mut max_n_turn = 0;
    for y in 0..h {
        for x in 0..w {
            if let Some((_, n)) = clues[y][x] {
                max_n_turn = max_n_turn.max(n);
            }
        }
    }
    let n_turn_unknown = max_n_turn + 1;
    let n_turn_horizontal = &solver.int_var_2d((h, w - 1), 0, max_n_turn + 1);
    let n_turn_vertical = &solver.int_var_2d((h - 1, w), 0, max_n_turn + 1);

    for y in 0..h {
        for x in 0..w {
            let mut adj: [Option<(BoolVar, BoolVar, IntVar)>; 4] = [None, None, None, None];

            if y > 0 {
                adj[0] = Some((
                    line_dr.vertical.at((y - 1, x)),
                    line_ul.vertical.at((y - 1, x)),
                    n_turn_vertical.at((y - 1, x)),
                ));
            }
            if y < h - 1 {
                adj[1] = Some((
                    line_ul.vertical.at((y, x)),
                    line_dr.vertical.at((y, x)),
                    n_turn_vertical.at((y, x)),
                ));
            }
            if x > 0 {
                adj[2] = Some((
                    line_dr.horizontal.at((y, x - 1)),
                    line_ul.horizontal.at((y, x - 1)),
                    n_turn_horizontal.at((y, x - 1)),
                ));
            }
            if x < w - 1 {
                adj[3] = Some((
                    line_ul.horizontal.at((y, x)),
                    line_dr.horizontal.at((y, x)),
                    n_turn_horizontal.at((y, x)),
                ));
            }

            if let Some((dir, n)) = clues[y][x] {
                let out_idx = match dir {
                    Arrow::Unspecified => panic!(),
                    Arrow::Up => 0,
                    Arrow::Down => 1,
                    Arrow::Left => 2,
                    Arrow::Right => 3,
                };
                if adj[out_idx].is_none() {
                    return None;
                }
                let (_, out_edge, n_turn) = adj[out_idx].as_ref().unwrap();
                solver.add_expr(out_edge);
                solver.add_expr(n_turn.eq(if n < 0 { n_turn_unknown } else { n }));

                for i in 0..4 {
                    if i == out_idx {
                        continue;
                    }
                    if let Some((in_edge_i, out_edge_i, n_turn_i)) = adj[i].as_ref() {
                        solver.add_expr(!out_edge_i);
                        solver
                            .add_expr(in_edge_i.imp(n_turn_i.eq(0) | n_turn_i.eq(n_turn_unknown)));
                    }
                }
            } else {
                let mut in_edges = vec![];
                let mut out_edges = vec![];
                for elm in &adj {
                    if let Some((in_edge, out_edge, _)) = elm {
                        in_edges.push(in_edge);
                        out_edges.push(out_edge);
                    }
                }
                solver.add_expr(count_true(&in_edges).le(1));
                solver.add_expr(count_true(&in_edges).eq(count_true(&out_edges)));

                for i in 0..4 {
                    for j in 0..4 {
                        if i != j {
                            if let (
                                Some((in_edge_i, _, n_turn_i)),
                                Some((_, out_edge_j, n_turn_j)),
                            ) = (&adj[i], &adj[j])
                            {
                                if i / 2 == j / 2 {
                                    solver.add_expr(
                                        (in_edge_i & out_edge_j).imp(n_turn_i.eq(n_turn_j)),
                                    );
                                } else {
                                    solver.add_expr((in_edge_i & out_edge_j).imp(
                                        (n_turn_i.eq(n_turn_unknown) & n_turn_j.eq(n_turn_unknown))
                                            | (n_turn_i.eq(n_turn_j + 1)),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<Option<NumberedArrow>>>;

fn combinator() -> impl Combinator<Problem> {
    MaybeSkip::new(
        "b/",
        Grid::new(Choice::new(vec![
            Box::new(Optionalize::new(NumberedArrowCombinator)),
            Box::new(Spaces::new(None, 'a')),
        ])),
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "firefly", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["firefly"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        let mut problem = vec![vec![None; 5]; 6];
        problem[1][1] = Some((Arrow::Up, -1));
        problem[1][3] = Some((Arrow::Right, 3));
        problem[2][1] = Some((Arrow::Right, -1));
        problem[2][3] = Some((Arrow::Right, 2));
        problem[3][1] = Some((Arrow::Down, -1));
        problem[3][3] = Some((Arrow::Left, 2));
        problem[5][1] = Some((Arrow::Left, -1));
        problem
    }

    #[test]
    fn test_firefly_problem() {
        let problem = problem_for_tests();
        let ans = solve_firefly(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_firefly_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?firefly/5/6/f1.a43b4.a42b2.a32g3.c";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
