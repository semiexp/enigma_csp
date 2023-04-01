use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, AlphaToNum, Choice, Combinator, Dict, Grid, HexInt, Map,
    Spaces, Tuple2,
};
use crate::solver::Solver;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ReflectLinkClue {
    None,
    UpperLeft(i32),
    UpperRight(i32),
    LowerLeft(i32),
    LowerRight(i32),
    Cross,
}

impl ReflectLinkClue {
    fn to_tuple(&self) -> (i32, i32) {
        match self {
            &ReflectLinkClue::UpperLeft(n) => (4, n),
            &ReflectLinkClue::UpperRight(n) => (3, n),
            &ReflectLinkClue::LowerLeft(n) => (1, n),
            &ReflectLinkClue::LowerRight(n) => (2, n),
            _ => (-1, -1),
        }
    }

    fn from_tuple(t: (i32, i32)) -> ReflectLinkClue {
        let (kind, n) = t;
        match kind {
            1 => ReflectLinkClue::LowerLeft(n),
            2 => ReflectLinkClue::LowerRight(n),
            3 => ReflectLinkClue::UpperRight(n),
            4 => ReflectLinkClue::UpperLeft(n),
            _ => panic!(),
        }
    }
}

pub fn solve_reflect_link(
    clues: &[Vec<ReflectLinkClue>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let (_, is_cross) = graph::crossable_single_cycle_grid_edges(&mut solver, &is_line);
    for y in 0..h {
        for x in 0..w {
            solver.add_expr(
                is_cross
                    .at((y, x))
                    .iff(clues[y][x] == ReflectLinkClue::Cross),
            );

            let (to_down, to_right, n) = match &clues[y][x] {
                &ReflectLinkClue::UpperLeft(n) => (true, true, n),
                &ReflectLinkClue::UpperRight(n) => (true, false, n),
                &ReflectLinkClue::LowerLeft(n) => (false, true, n),
                &ReflectLinkClue::LowerRight(n) => (false, false, n),
                _ => continue,
            };

            let n_vertical = if to_down {
                is_line
                    .vertical
                    .slice_fixed_x((y.., x))
                    .consecutive_prefix_true()
            } else {
                is_line
                    .vertical
                    .slice_fixed_x((..y, x))
                    .reverse()
                    .consecutive_prefix_true()
            };
            let n_horizontal = if to_right {
                is_line
                    .horizontal
                    .slice_fixed_y((y, x..))
                    .consecutive_prefix_true()
            } else {
                is_line
                    .horizontal
                    .slice_fixed_y((y, ..x))
                    .reverse()
                    .consecutive_prefix_true()
            };
            solver.add_expr(n_vertical.gt(0));
            solver.add_expr(n_horizontal.gt(0));
            if n > 0 {
                solver.add_expr((n_vertical + n_horizontal).eq(n - 1));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<ReflectLinkClue>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Spaces::new(ReflectLinkClue::None, 'a')),
        Box::new(Dict::new(ReflectLinkClue::Cross, "5")),
        Box::new(Map::new(
            Tuple2::new(AlphaToNum::new('1', '4', 1), HexInt),
            |x: ReflectLinkClue| Some(x.to_tuple()),
            |x| Some(ReflectLinkClue::from_tuple(x)),
        )),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "reflect", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["reflect"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        let mut ret = vec![vec![ReflectLinkClue::None; 5]; 6];
        ret[0][0] = ReflectLinkClue::UpperLeft(0);
        ret[1][2] = ReflectLinkClue::UpperRight(0);
        ret[2][1] = ReflectLinkClue::Cross;
        ret[3][1] = ReflectLinkClue::Cross;
        ret[3][3] = ReflectLinkClue::LowerRight(6);
        ret[4][0] = ReflectLinkClue::LowerLeft(5);
        ret[4][1] = ReflectLinkClue::Cross;
        ret
    }

    #[test]
    fn test_reflect_link_problem() {
        let problem = problem_for_tests();
        let ans = solve_reflect_link(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::GridEdges {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 1, 1],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_reflect_link_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?reflect/5/6/40f30c5d5a26a155h";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
