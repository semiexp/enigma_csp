use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{any, Solver};

pub fn solve_shikaku(
    clues: &[Vec<Option<i32>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let edges = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&edges.horizontal);
    solver.add_answer_key_bool(&edges.vertical);

    for y in 1..h {
        for x in 1..w {
            solver.add_expr(
                !((edges.horizontal.at((y - 1, x - 1)) ^ edges.horizontal.at((y - 1, x)))
                    & (edges.vertical.at((y - 1, x - 1)) ^ edges.vertical.at((y, x - 1)))),
            );
        }
    }

    let mut clue_pos = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                clue_pos.push((y, x, n));
            }
        }
    }

    if clue_pos.len() == 0 {
        return None;
    }

    let ids = solver.int_var_2d((h, w), 0, clue_pos.len() as i32 - 1);
    for i in 0..clue_pos.len() {
        graph::active_vertices_connected_2d(&mut solver, ids.eq(i as i32));
        let (y, x, n) = clue_pos[i];
        solver.add_expr(ids.at((y, x)).eq(i as i32));
        if n > 0 {
            let rect_up = (!edges.horizontal.slice_fixed_x((..y, x)))
                .reverse()
                .consecutive_prefix_true();
            let rect_down = (!edges.horizontal.slice_fixed_x((y.., x))).consecutive_prefix_true();
            let rect_height = rect_up + rect_down + 1;

            let rect_left = (!edges.vertical.slice_fixed_y((y, ..x)))
                .reverse()
                .consecutive_prefix_true();
            let rect_right = (!edges.vertical.slice_fixed_y((y, x..))).consecutive_prefix_true();
            let rect_width = rect_left + rect_right + 1;

            let mut cand = vec![];
            for a in 1..=n {
                if n % a == 0 {
                    let b = n / a;
                    cand.push(rect_height.eq(a) & rect_width.eq(b));
                }
            }
            solver.add_expr(any(cand));
            //solver.add_expr(ids.eq(i as i32).count_true().eq(n));
        }
    }
    solver.add_expr(
        edges
            .horizontal
            .iff(ids.slice((..(h - 1), ..)).ne(ids.slice((1.., ..)))),
    );
    solver.add_expr(
        edges
            .vertical
            .iff(ids.slice((.., ..(w - 1))).ne(ids.slice((.., 1..)))),
    );

    solver.irrefutable_facts().map(|f| f.get(edges))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "shikaku", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["shikaku"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, Some(6), Some(6), None, None, None],
            vec![None, Some(4), Some(-1), None, Some(8), None],
            vec![None, None, None, None, None, None],
            vec![None, Some(4), None, None, Some(4), None],
        ]
    }

    #[test]
    fn test_shikaku_problem() {
        let problem = problem_for_tests();
        let ans = solve_shikaku(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        #[rustfmt::skip]
        let expected = graph::InnerGridEdges {
            horizontal: vec![
                vec![Some(false), Some(false), Some(false), Some(false), Some(false), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(false), Some(false)],
                vec![Some(true), Some(true), Some(true), Some(true), Some(false), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(false), Some(true), Some(true)],
                vec![Some(true), Some(true), Some(true), Some(true), Some(false), Some(false)],
            ],
            vertical: vec![
                vec![Some(false), Some(true), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(true), Some(false)],
            ],
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_shikaku_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?shikaku/6/6/s66j4.g8n4h4g";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
