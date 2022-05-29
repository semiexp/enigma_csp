use super::util;
use crate::graph;
use crate::serializer::{problem_to_url, url_to_problem, Combinator, Grid, Map, MultiDigit};
use crate::solver::Solver;

pub fn solve_simpleloop(is_black: &[Vec<bool>]) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(is_black);

    let mut parity_diff = 0;
    for y in 0..h {
        for x in 0..w {
            if !is_black[y][x] {
                if (y + x) % 2 == 0 {
                    parity_diff += 1;
                } else {
                    parity_diff -= 1;
                }
            }
        }
    }
    if parity_diff != 0 {
        return None;
    }
    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_edges(&mut solver, &is_line);

    for y in 0..h {
        for x in 0..w {
            solver.add_expr(is_passed.at((y, x)) ^ is_black[y][x]);
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<bool>>;

fn combinator() -> impl Combinator<Vec<Vec<bool>>> {
    Grid::new(Map::new(
        MultiDigit::new(2, 5),
        |x: bool| Some(if x { 1 } else { 0 }),
        |n: i32| Some(n == 1),
    ))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "simpleloop", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["simpleloop"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Vec<Vec<bool>> {
        // https://puzsq.jp/main/puzzle_play.php?pid=9833
        let mut ret = vec![vec![false; 8]; 7];
        ret[0][3] = true;
        ret[2][2] = true;
        ret[3][7] = true;
        ret[4][1] = true;
        ret[4][5] = true;
        ret[5][3] = true;
        ret
    }

    #[test]
    fn test_simpleloop_problem() {
        let problem = problem_for_tests();
        let ans = solve_simpleloop(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        for y in 0..problem.len() {
            for x in 0..problem[0].len() {
                if y + 1 < problem.len() {
                    assert!(ans.vertical[y][x].is_some());
                }
                if x + 1 < problem[0].len() {
                    assert!(ans.horizontal[y][x].is_some());
                }
            }
        }
        assert_eq!(ans.horizontal[3][1], Some(true));
        assert_eq!(ans.horizontal[3][2], Some(false));
    }

    #[test]
    fn test_simpleloop_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?simpleloop/8/7/200200a42000";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
