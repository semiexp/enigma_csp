use super::util;
use crate::graph;
use crate::solver::{all, any, count_true, Solver, FALSE};

const EIGHT_NEIGHBORS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
];

pub fn solve_tapa(clues: &[Vec<Option<[i32; 4]>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    graph::active_vertices_connected_2d(&mut solver, is_black);

    solver.add_expr(
        !(is_black.slice((..(h - 1), ..(w - 1)))
            & is_black.slice((..(h - 1), 1..))
            & is_black.slice((1.., ..(w - 1)))
            & is_black.slice((1.., 1..))),
    );

    for y in 0..h {
        for x in 0..w {
            if let Some(clue) = clues[y][x] {
                solver.add_expr(!is_black.at((y, x)));

                let mut neighbors = vec![];
                for &(dy, dx) in &EIGHT_NEIGHBORS {
                    let y2 = y as i32 + dy;
                    let x2 = x as i32 + dx;
                    if 0 <= y2 && y2 < h as i32 && 0 <= x2 && x2 < w as i32 {
                        neighbors.push(is_black.at((y2 as usize, x2 as usize)).expr());
                    } else {
                        neighbors.push(FALSE);
                    }
                }

                if clue[0] == -1 || clue[0] == 0 {
                    solver.add_expr(!any(&neighbors));
                    continue;
                }
                if clue[0] == 8 {
                    solver.add_expr(all(&neighbors));
                    continue;
                }

                let mut clue_counts = [0; 9];
                let mut total_clue_counts = 0;
                let mut has_any = false;
                for i in 0..4 {
                    if clue[i] != -1 {
                        assert!(clue[i] == -2 || 0 <= clue[i] && clue[i] <= 7);
                        if clue[i] == -2 {
                            has_any = true;
                        } else {
                            clue_counts[clue[i] as usize] += 1;
                        }
                        total_clue_counts += 1;
                    }
                }

                for l in 1..=8 {
                    if clue_counts[l] == 0 {
                        continue;
                    }
                    let mut conds = vec![];
                    for s in 0..8 {
                        let mut cond = vec![
                            !(neighbors[s].clone()),
                            !(neighbors[(s + l + 1) % 8].clone()),
                        ];
                        for i in 0..l {
                            cond.push(neighbors[(s + i + 1) % 8].clone());
                        }
                        conds.push(all(cond));
                    }
                    if has_any {
                        solver.add_expr(count_true(conds).ge(clue_counts[l]));
                    } else {
                        solver.add_expr(count_true(conds).eq(clue_counts[l]));
                    }
                }

                let mut unit_count = vec![];
                for s in 0..8 {
                    unit_count.push(&neighbors[s] & !&neighbors[(s + 1) % 8]);
                }
                solver.add_expr(count_true(unit_count).eq(total_clue_counts));
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

pub type Problem = Vec<Vec<Option<[i32; 4]>>>;

#[cfg(test)]
mod tests {
    pub use super::*;

    fn problem_for_tests1() -> Problem {
        let height = 6;
        let width = 7;
        let mut ret: Problem = vec![vec![None; width]; height];

        ret[0][0] = Some([2, -1, -1, -1]);
        ret[1][2] = Some([1, 5, -1, -1]);
        ret[1][4] = Some([1, 1, 1, 1]);
        ret[4][1] = Some([8, -1, -1, -1]);
        ret[5][4] = Some([0, -1, -1, -1]);

        ret
    }

    fn problem_for_tests2() -> Problem {
        let height = 8;
        let width = 6;
        let mut ret: Problem = vec![vec![None; width]; height];

        ret[1][5] = Some([2, -1, -1, -1]);
        ret[2][1] = Some([1, 1, 1, 1]);
        ret[2][3] = Some([-2, -1, -1, -1]);
        ret[4][3] = Some([-2, -2, -2, -1]);
        ret[6][2] = Some([-2, -2, -1, -1]);
        ret[6][3] = Some([3, -2, -2, -1]);

        ret
    }

    #[test]
    fn test_tapa_problem1() {
        let problem = problem_for_tests1();
        let ans = solve_tapa(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = [
            [0, 1, 1, 1, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ];
        for y in 0..6 {
            for x in 0..7 {
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
    fn test_tapa_problem2() {
        let problem = problem_for_tests2();
        let ans = solve_tapa(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        let expected = [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ];
        for y in 0..8 {
            for x in 0..6 {
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
