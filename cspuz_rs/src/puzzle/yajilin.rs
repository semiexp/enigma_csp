use crate::graph;
use crate::solver::Solver;

#[derive(Clone, Copy)]
pub enum YajilinClue {
    Unspecified(i32),
    Up(i32),
    Down(i32),
    Left(i32),
    Right(i32),
}

pub fn solve_yajilin(
    clues: &[Vec<Option<YajilinClue>>],
) -> Option<(graph::BoolGridFrameIrrefutableFacts, Vec<Vec<Option<bool>>>)> {
    let h = clues.len();
    assert!(h > 0);
    let w = clues[0].len();

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridFrame::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_frame(&mut solver, is_line);
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);
    solver.add_expr(!(is_black.slice((..(h - 1), ..)) & is_black.slice((1.., ..))));
    solver.add_expr(!(is_black.slice((.., ..(w - 1))) & is_black.slice((.., 1..))));

    for y in 0..h {
        for x in 0..w {
            if let Some(clue) = clues[y][x] {
                solver.add_expr(!is_passed.at((y, x)));
                solver.add_expr(!is_black.at((y, x)));

                match clue {
                    YajilinClue::Unspecified(_) => unimplemented!(),
                    YajilinClue::Up(n) => {
                        if n >= 0 {
                            solver.add_expr(is_black.slice_fixed_x((..y, x)).count_true().eq(n));
                        }
                    }
                    YajilinClue::Down(n) => {
                        if n >= 0 {
                            solver.add_expr(
                                is_black.slice_fixed_x(((y + 1).., x)).count_true().eq(n),
                            );
                        }
                    }
                    YajilinClue::Left(n) => {
                        if n >= 0 {
                            solver.add_expr(is_black.slice_fixed_y((y, ..x)).count_true().eq(n));
                        }
                    }
                    YajilinClue::Right(n) => {
                        if n >= 0 {
                            solver.add_expr(
                                is_black.slice_fixed_y((y, (x + 1)..)).count_true().eq(n),
                            );
                        }
                    }
                }
            } else {
                solver.add_expr(is_passed.at((y, x)) ^ is_black.at((y, x)));
            }
        }
    }

    solver
        .irrefutable_facts()
        .map(|f| (f.get(is_line), f.get(is_black)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yajilin_problem() {
        // https://puzsq.jp/main/puzzle_play.php?pid=8218
        let mut problem = vec![vec![None; 10]; 10];
        problem[2][3] = Some(YajilinClue::Left(2));
        problem[2][5] = Some(YajilinClue::Right(1));
        problem[2][8] = Some(YajilinClue::Down(1));
        problem[3][0] = Some(YajilinClue::Down(1));
        problem[4][3] = Some(YajilinClue::Down(2));
        problem[4][9] = Some(YajilinClue::Left(0));
        problem[6][3] = Some(YajilinClue::Down(1));
        problem[6][5] = Some(YajilinClue::Up(2));
        problem[6][8] = Some(YajilinClue::Up(1));
        problem[8][7] = Some(YajilinClue::Down(0));
        problem[9][2] = Some(YajilinClue::Left(0));

        let ans = solve_yajilin(&problem);
        assert!(ans.is_some());
        let (_, is_black) = ans.unwrap();

        let expected_base = [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        ];
        let expected =
            expected_base.map(|row| row.iter().map(|&n| Some(n == 1)).collect::<Vec<_>>());
        assert_eq!(is_black, expected);
    }
}
