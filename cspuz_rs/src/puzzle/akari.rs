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
}
