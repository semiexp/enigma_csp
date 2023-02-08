use crate::solver::Solver;

pub fn solve_star_battle(
    n: usize,
    k: i32,
    rooms: &[Vec<(usize, usize)>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let mut solver = Solver::new();
    let has_star = solver.bool_var_2d((n, n));
    solver.add_answer_key_bool(&has_star);

    for i in 0..n {
        solver.add_expr(has_star.slice_fixed_y((i, ..)).count_true().eq(k));
        solver.add_expr(has_star.slice_fixed_x((.., i)).count_true().eq(k));
    }
    solver.add_expr(!(has_star.slice((..(n - 1), ..)) & has_star.slice((1.., ..))));
    solver.add_expr(!(has_star.slice((.., ..(n - 1))) & has_star.slice((.., 1..))));
    solver.add_expr(!(has_star.slice((..(n - 1), ..(n - 1))) & has_star.slice((1.., 1..))));
    solver.add_expr(!(has_star.slice((..(n - 1), 1..)) & has_star.slice((1.., ..(n - 1)))));

    for room in rooms {
        solver.add_expr(has_star.select(room).count_true().eq(k));
    }

    solver.irrefutable_facts().map(|f| f.get(&has_star))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_star_battle_problem() {
        let n = 6;
        let k = 1;
        let rooms = [
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3)],
            vec![(0, 4), (0, 5), (1, 4), (1, 5), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 3)],
            vec![(1, 1), (2, 0), (2, 1), (3, 0), (4, 0), (5, 0), (5, 1)],
            vec![(1, 2), (2, 2), (2, 3), (2, 4)],
            vec![(4, 1), (4, 2), (4, 4), (5, 2), (5, 3), (5, 4)],
            vec![(4, 5), (5, 5)],
        ];
        let ans = solve_star_battle(n, k, &rooms);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ]);
        assert_eq!(ans, expected);
    }
}
