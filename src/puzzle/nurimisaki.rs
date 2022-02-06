use super::graph;
use super::solver::{any, Solver};

pub fn solve_nurimisaki(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let h = clues.len();
    assert!(h > 0);
    let w = clues[0].len();

    let mut solver = Solver::new();
    let is_white = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_white);

    graph::active_vertices_connected_2d(&mut solver, is_white);

    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            solver.add_expr(is_white.slice((y..(y + 2), x..(x + 2))).any());
            solver.add_expr(!is_white.slice((y..(y + 2), x..(x + 2))).all());
        }
    }
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                solver.add_expr(is_white.at((y, x)));
                solver.add_expr(is_white.four_neighbors((y, x)).count_true().eq(1));
                if n > 0 {
                    let n = n as usize;
                    let mut dirs = vec![];
                    if y >= n - 1 {
                        if y == n - 1 {
                            dirs.push(is_white.slice_fixed_x(((y - (n - 1))..y, x)).all());
                        } else {
                            dirs.push(
                                is_white.slice_fixed_x(((y - (n - 1))..y, x)).all()
                                    & !is_white.at((y - n, x)),
                            );
                        }
                    }
                    if x >= n - 1 {
                        if x == n - 1 {
                            dirs.push(is_white.slice_fixed_y((y, (x - (n - 1))..x)).all());
                        } else {
                            dirs.push(
                                is_white.slice_fixed_y((y, (x - (n - 1))..x)).all()
                                    & !is_white.at((y, x - n)),
                            );
                        }
                    }
                    if h - y >= n {
                        dirs.push(
                            is_white.slice_fixed_x(((y + 1)..(y + n), x)).all()
                                & !is_white.at_or((y + n, x), false),
                        );
                    }
                    if w - x >= n {
                        dirs.push(
                            is_white.slice_fixed_y((y, (x + 1)..(x + n))).all()
                                & !is_white.at_or((y, x + n), false),
                        );
                    }
                    solver.add_expr(any(dirs));
                }
            } else {
                solver.add_expr(
                    is_white
                        .at((y, x))
                        .imp(is_white.four_neighbors((y, x)).count_true().ne(1)),
                );
            }
        }
    }
    solver.irrefutable_facts().map(|f| f.get(is_white))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_nurimisaki_problem() {
        // https://twitter.com/semiexp/status/1168898897424633856
        let problem = [
            vec![None, None, None, None, Some(3), None, None, None, None, None],
            vec![None, Some(3), None, None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, None, Some(2), None],
            vec![None, None, None, None, None, None, None, None, None, None],
            vec![None, None, None, Some(2), None, None, None, None, None, None],
            vec![None, None, None, None, Some(0), None, Some(2), None, None, None],
            vec![None, Some(2), None, None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, None, None, None, Some(2)],
            vec![None, None, None, None, None, Some(2), None, None, None, None],
            vec![None, None, None, None, Some(3), None, None, None, None, None],
        ];
        let ans = solve_nurimisaki(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = [
            [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        ];
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(ans[y][x], Some(expected[y][x] == 0), "mismatch at ({}, {})", y, x);
            }
        }
    }
}
