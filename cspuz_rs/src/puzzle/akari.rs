use crate::serializer::{
    from_base16, problem_to_url, to_base16, url_to_problem, Choice, Combinator, Context, Dict,
    Grid, Spaces,
};
use crate::solver::{BoolVar, Solver};

pub fn solve_akari(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let h = clues.len();
    assert!(h > 0);
    let w = clues[0].len();

    let mut solver = Solver::new();
    let has_light = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(has_light);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(!has_light.at((y, x)));
                if n >= 0 {
                    solver.add_expr(has_light.four_neighbors((y, x)).count_true().eq(n));
                }
            }
        }
    }

    let mut horizontal_group: Vec<Vec<Option<BoolVar>>> = vec![vec![None; w]; h];
    for y in 0..h {
        let mut start: Option<usize> = None;
        for x in 0..=w {
            if x < w && clues[y][x].is_none() {
                if start.is_none() {
                    start = Some(x);
                }
            } else {
                if let Some(s) = start {
                    let v = solver.bool_var();
                    solver.add_expr(
                        has_light
                            .slice_fixed_y((y, s..x))
                            .count_true()
                            .eq(v.ite(1, 0)),
                    );
                    for x2 in s..x {
                        horizontal_group[y][x2] = Some(v.clone());
                        println!("{} {}", y, x2);
                    }
                    println!();
                    start = None;
                }
            }
        }
    }

    let mut vertical_group: Vec<Vec<Option<BoolVar>>> = vec![vec![None; w]; h];
    for x in 0..w {
        let mut start: Option<usize> = None;
        for y in 0..=h {
            if y < h && clues[y][x].is_none() {
                if start.is_none() {
                    start = Some(y);
                }
            } else {
                if let Some(s) = start {
                    let v = solver.bool_var();
                    solver.add_expr(
                        has_light
                            .slice_fixed_x((s..y, x))
                            .count_true()
                            .eq(v.ite(1, 0)),
                    );
                    for y2 in s..y {
                        vertical_group[y2][x] = Some(v.clone());
                    }
                    start = None;
                }
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if clues[y][x].is_none() {
                solver.add_expr(
                    horizontal_group[y][x].as_ref().unwrap()
                        | vertical_group[y][x].as_ref().unwrap(),
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(has_light))
}

struct AkariClueCombinator;

impl Combinator<Option<i32>> for AkariClueCombinator {
    fn serialize(&self, _: &Context, input: &[Option<i32>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let n = input[0]?;
        if n < 0 {
            return None;
        }
        let mut n_spaces = 0;
        while n_spaces < 2 && 1 + n_spaces < input.len() && input[1 + n_spaces].is_none() {
            n_spaces += 1;
        }
        Some((1 + n_spaces, vec![to_base16(n + n_spaces as i32 * 5)]))
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<Option<i32>>)> {
        if input.len() == 0 {
            return None;
        }
        let c = from_base16(input[0])?;
        if c == 15 {
            return None;
        }
        let n = c % 5;
        let spaces = c / 5;
        let mut ret = vec![Some(n)];
        for _ in 0..spaces {
            ret.push(None);
        }
        Some((1, ret))
    }
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(AkariClueCombinator),
        Box::new(Spaces::new(None, 'g')),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "akari", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["akari"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_akari_problem() {
        // https://twitter.com/semiexp/status/1225770511080144896
        let problem = [
            vec![None, None, Some(2), None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, None, Some(2), None],
            vec![None, None, None, None, None, None, None, Some(-1), None, None],
            vec![Some(-1), None, None, None, Some(3), None, None, None, None, None],
            vec![None, None, None, None, None, Some(-1), None, None, None, Some(-1)],
            vec![Some(2), None, None, None, Some(2), None, None, None, None, None],
            vec![None, None, None, None, None, Some(3), None, None, None, Some(-1)],
            vec![None, None, Some(-1), None, None, None, None, None, None, None],
            vec![None, Some(2), None, None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, Some(-1), None, None],
        ];
        let ans = solve_akari(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        println!("{:?}", ans);
        let expected = [
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        ];
        for y in 0..10 {
            for x in 0..10 {
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
    #[rustfmt::skip]
    fn test_akari_serializer() {
        let problem = vec![
            vec![None, None, Some(2), None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, None, Some(2), None],
            vec![None, None, None, None, None, None, None, Some(-1), None, None],
            vec![Some(-1), None, None, None, Some(3), None, None, None, None, None],
            vec![None, None, None, None, None, Some(-1), None, None, None, Some(-1)],
            vec![Some(2), None, None, None, Some(2), None, None, None, None, None],
            vec![None, None, None, None, None, Some(3), None, None, None, Some(-1)],
            vec![None, None, Some(-1), None, None, None, None, None, None, None],
            vec![None, Some(2), None, None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, Some(-1), None, None],
        ];
        let url = "https://puzz.link/p?akari/10/10/hcscl.h.idn.i.cgcndg.h.ncs.h";

        let deserialized = deserialize_problem(url);
        assert!(deserialized.is_some());
        let deserialized = deserialized.unwrap();
        assert_eq!(problem, deserialized);
        let reserialized = serialize_problem(&deserialized);
        assert!(reserialized.is_some());
        let reserialized = reserialized.unwrap();
        assert_eq!(reserialized, url);
    }
}
