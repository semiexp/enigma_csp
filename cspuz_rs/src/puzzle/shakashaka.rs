use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, NumSpaces, Spaces,
};
use crate::solver::{count_true, Solver, FALSE};

#[derive(PartialEq, Eq, Debug)]
pub enum ShakashakaCell {
    Blank,
    UpperLeft,
    LowerLeft,
    LowerRight,
    UpperRight,
}

pub fn solve_shakashaka(problem: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<ShakashakaCell>>>> {
    let h = problem.len();
    assert!(h > 0);
    let w = problem[0].len();

    // 1   2   3   4
    // +-+ +     + +-+
    // |/  |\   /|  \|
    // +   +-+ +-+   +
    let mut solver = Solver::new();
    let ans = &solver.int_var_2d((h, w), 0, 4);
    solver.add_answer_key_int(ans);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = problem[y][x] {
                solver.add_expr(ans.at((y, x)).eq(0));
                if n >= 0 {
                    solver.add_expr(ans.four_neighbors((y, x)).ne(0).count_true().eq(n));
                }
            }
        }
    }
    for y in 0..=h {
        for x in 0..=w {
            let mut diagonals = vec![];
            let mut is_empty = vec![];
            let mut is_white_angle = vec![];

            if y > 0 && x > 0 {
                let a = &ans.at((y - 1, x - 1));
                diagonals.push(a.eq(4));
                diagonals.push(a.eq(2));
                if problem[y - 1][x - 1].is_none() {
                    is_empty.push(a.eq(0));
                    is_white_angle.push(a.eq(0) | a.eq(1));
                } else {
                    is_empty.push(FALSE);
                }
            } else {
                diagonals.push(FALSE);
                diagonals.push(FALSE);
                is_empty.push(FALSE);
            }
            if y < h && x > 0 {
                let a = &ans.at((y, x - 1));
                diagonals.push(a.eq(1));
                diagonals.push(a.eq(3));
                if problem[y][x - 1].is_none() {
                    is_empty.push(a.eq(0));
                    is_white_angle.push(a.eq(0) | a.eq(2));
                } else {
                    is_empty.push(FALSE);
                }
            } else {
                diagonals.push(FALSE);
                diagonals.push(FALSE);
                is_empty.push(FALSE);
            }
            if y < h && x < w {
                let a = &ans.at((y, x));
                diagonals.push(a.eq(2));
                diagonals.push(a.eq(4));
                if problem[y][x].is_none() {
                    is_empty.push(a.eq(0));
                    is_white_angle.push(a.eq(0) | a.eq(3));
                } else {
                    is_empty.push(FALSE);
                }
            } else {
                diagonals.push(FALSE);
                diagonals.push(FALSE);
                is_empty.push(FALSE);
            }
            if y > 0 && x < w {
                let a = &ans.at((y - 1, x));
                diagonals.push(a.eq(3));
                diagonals.push(a.eq(1));
                if problem[y - 1][x].is_none() {
                    is_empty.push(a.eq(0));
                    is_white_angle.push(a.eq(0) | a.eq(4));
                } else {
                    is_empty.push(FALSE);
                }
            } else {
                diagonals.push(FALSE);
                diagonals.push(FALSE);
                is_empty.push(FALSE);
            }

            for i in 0..8 {
                if i % 2 == 0 {
                    solver.add_expr(diagonals[i].imp(
                        &diagonals[(i + 3) % 8]
                            | (&is_empty[(i + 3) % 8 / 2] & &diagonals[(i + 5) % 8]),
                    ));
                } else {
                    solver.add_expr(diagonals[i].imp(
                        &diagonals[(i + 5) % 8]
                            | (&is_empty[(i + 5) % 8 / 2] & &diagonals[(i + 3) % 8]),
                    ));
                }
            }
            solver.add_expr(count_true(is_white_angle).ne(3));
        }
    }

    solver.irrefutable_facts().map(|f| {
        let model = f.get(ans);
        model
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|v| {
                        v.map(|n| match n {
                            0 => ShakashakaCell::Blank,
                            1 => ShakashakaCell::UpperLeft,
                            2 => ShakashakaCell::LowerLeft,
                            3 => ShakashakaCell::LowerRight,
                            4 => ShakashakaCell::UpperRight,
                            _ => panic!(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    })
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Spaces::new(None, 'g')),
        Box::new(NumSpaces::new(4, 2)),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "shakashaka", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["shakashaka"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Vec<Vec<Option<i32>>> {
        // https://twitter.com/semiexp/status/1223794016593956864
        let height = 10;
        let width = 10;
        let mut problem = vec![vec![None; width]; height];
        problem[1][2] = Some(3);
        problem[2][7] = Some(2);
        problem[2][9] = Some(0);
        problem[3][0] = Some(1);
        problem[3][3] = Some(3);
        problem[4][6] = Some(3);
        problem[5][0] = Some(2);
        problem[5][3] = Some(2);
        problem[6][8] = Some(2);
        problem[9][3] = Some(2);
        problem[9][7] = Some(0);
        problem
    }

    #[test]
    fn test_shakashaka_problem() {
        let problem = problem_for_tests();
        let ans = solve_shakashaka(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        for y in 0..ans.len() {
            for x in 0..ans[0].len() {
                assert!(ans[y][x].is_some());
            }
        }
        assert_eq!(ans[0][5], Some(ShakashakaCell::UpperLeft));
        assert_eq!(ans[7][4], Some(ShakashakaCell::UpperRight));
        assert_eq!(ans[6][2], Some(ShakashakaCell::LowerRight));
    }

    #[test]
    fn test_shakashaka_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?shakashaka/10/10/rdr70bdpdgccrczhcga";

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
