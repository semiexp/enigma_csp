use crate::graph;
use crate::solver::{count_true, Solver};

pub enum GateDir {
    Horizontal,
    Vertical,
}

pub struct Gate {
    cells: Vec<(usize, usize)>,
    ord: Option<i32>,
}

pub fn solve_slalom(
    origin: (usize, usize),
    is_black: &[Vec<bool>],
    gates: &[Gate],
) -> Option<graph::BoolGridFrameIrrefutableFacts> {
    let h = is_black.len();
    assert!(h > 0);
    let w = is_black[0].len();

    let mut solver = Solver::new();
    let line = &graph::BoolGridFrame::new(&mut solver, (h - 1, w - 1));
    let line_dir = &graph::BoolGridFrame::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&line.horizontal);
    solver.add_answer_key_bool(&line.vertical);

    let passed = &graph::single_cycle_grid_frame(&mut solver, &line);
    let gate_ord = &solver.int_var_2d((h, w), 0, gates.len() as i32);

    let mut gate_id: Vec<Vec<Option<usize>>> = vec![vec![None; w]; h];
    for (i, gate) in gates.iter().enumerate() {
        let mut cells = vec![];
        for &(y, x) in &gate.cells {
            assert!(gate_id[y][x].is_none());
            gate_id[y][x] = Some(i);
            cells.push(passed.at((y, x)));
        }
        solver.add_expr(count_true(cells).eq(1));
    }
    solver.add_expr(passed.at(origin));

    for y in 0..h {
        for x in 0..w {
            let neighbors = passed.four_neighbor_indices((y, x));
            solver.add_expr(
                count_true(
                    neighbors
                        .iter()
                        .map(|&(y2, x2)| {
                            line.at((y + y2, x + x2))
                                & (line_dir.at((y + y2, x + x2)) ^ ((y2, x2) < (y, x)))
                        })
                        .collect::<Vec<_>>(),
                )
                .eq(passed.at((y, x)).ite(1, 0)),
            );
            solver.add_expr(
                count_true(
                    neighbors
                        .iter()
                        .map(|&(y2, x2)| {
                            line.at((y + y2, x + x2))
                                & (line_dir.at((y + y2, x + x2)).iff((y2, x2) < (y, x)))
                        })
                        .collect::<Vec<_>>(),
                )
                .eq(passed.at((y, x)).ite(1, 0)),
            );
            if is_black[y][x] {
                solver.add_expr(!passed.at((y, x)));
            }
            if (y, x) == origin {
                continue;
            }
            if let Some(g) = gate_id[y][x] {
                for (y2, x2) in neighbors {
                    solver.add_expr(
                        (line.at((y + y2, x + x2))
                            & (line_dir.at((y + y2, x + x2)) ^ ((y2, x2) < (y, x))))
                            .imp(gate_ord.at((y2, x2)).eq(gate_ord.at((y, x)) - 1)),
                    );
                }
                if let Some(n) = gates[g].ord {
                    solver.add_expr(passed.at((y, x)).imp(gate_ord.at((y, x)).eq(n)));
                }
            } else {
                for (y2, x2) in neighbors {
                    solver.add_expr(
                        (line.at((y + y2, x + x2))
                            & (line_dir.at((y + y2, x + x2)) ^ ((y2, x2) < (y, x))))
                            .imp(gate_ord.at((y2, x2)).eq(gate_ord.at((y, x)))),
                    );
                }
            }
        }
    }

    // auxiliary constraint
    for y0 in 0..h {
        for x0 in 0..w {
            for y1 in 0..h {
                for x1 in 0..w {
                    if (y0, x0) < (y1, x1) && gate_id[y0][x0].is_some() && gate_id[y1][x1].is_some()
                    {
                        solver.add_expr(
                            (passed.at((y0, x0)) & passed.at((y1, x1)))
                                .imp(gate_ord.at((y0, x0)).ne(gate_ord.at((y1, x1)))),
                        );
                    }
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(line))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slalom_problem() {
        // https://puzsq.jp/main/puzzle_play.php?pid=9522
        let is_black_base = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ];
        let is_black = is_black_base
            .iter()
            .map(|row| row.iter().map(|&n| n == 1).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let gates = vec![
            Gate {
                cells: vec![(1, 5), (1, 6), (1, 7)],
                ord: None,
            },
            Gate {
                cells: vec![(2, 3)],
                ord: None,
            },
            Gate {
                cells: vec![(3, 8)],
                ord: Some(1),
            },
            Gate {
                cells: vec![(6, 3), (6, 4), (6, 5), (6, 6)],
                ord: Some(3),
            },
            Gate {
                cells: vec![(7, 1)],
                ord: None,
            },
            Gate {
                cells: vec![(8, 6), (8, 7), (8, 8), (8, 9)],
                ord: Some(2),
            },
        ];

        let ans = solve_slalom((5, 1), &is_black, &gates);
        assert!(ans.is_some());
    }
}
