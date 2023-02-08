use super::util;
use crate::graph;
use crate::serializer::{
    url_to_problem, Choice, Combinator, Context, ContextBasedGrid, DecInt, Dict, FixedLengthHexInt,
    MaybeSkip, Seq, Sequencer, Size, Spaces, Tuple2,
};
use crate::solver::{count_true, Solver};

pub enum GateDir {
    Horizontal,
    Vertical,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Gate {
    cells: Vec<(usize, usize)>,
    ord: Option<i32>,
}

pub fn solve_slalom(
    origin: (usize, usize),
    is_black: &[Vec<bool>],
    gates: &[Gate],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(is_black);

    let mut solver = Solver::new();
    let line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    let line_dir = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&line.horizontal);
    solver.add_answer_key_bool(&line.vertical);

    let passed = &graph::single_cycle_grid_edges(&mut solver, &line);
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SlalomBlackCellDir {
    NoClue,
    NoDir,
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SlalomCell {
    White,
    Black(SlalomBlackCellDir, i32),
    Vertical,
    Horizontal,
}

struct SlalomAuxCombinator;

impl Combinator<Vec<Vec<SlalomCell>>> for SlalomAuxCombinator {
    fn serialize(&self, _: &Context, _: &[Vec<Vec<SlalomCell>>]) -> Option<(usize, Vec<u8>)> {
        unimplemented!();
    }

    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<Vec<Vec<SlalomCell>>>)> {
        let mut sequencer = Sequencer::new(input);
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let grid_combinator = ContextBasedGrid::new(Choice::new(vec![
            Box::new(Dict::new(
                SlalomCell::Black(SlalomBlackCellDir::NoClue, -1),
                "1",
            )),
            Box::new(Dict::new(SlalomCell::Vertical, "2")),
            Box::new(Dict::new(SlalomCell::Horizontal, "3")),
            Box::new(Spaces::new(SlalomCell::White, '4')),
        ]));
        let mut grid: Vec<Vec<SlalomCell>> =
            sequencer.deserialize_one_elem(ctx, grid_combinator)?;

        let mut n_black = 0usize;
        for y in 0..height {
            for x in 0..width {
                if let SlalomCell::Black(_, _) = grid[y][x] {
                    n_black += 1;
                }
            }
        }

        let seq_combinator = Seq::new(
            Choice::new(vec![
                Box::new(Spaces::new((SlalomBlackCellDir::NoClue, -1), 'g')),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::NoDir, "0"),
                    FixedLengthHexInt::new(1),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Up, "1"),
                    FixedLengthHexInt::new(1),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Down, "2"),
                    FixedLengthHexInt::new(1),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Left, "3"),
                    FixedLengthHexInt::new(1),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Right, "4"),
                    FixedLengthHexInt::new(1),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::NoDir, "5"),
                    FixedLengthHexInt::new(2),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Up, "6"),
                    FixedLengthHexInt::new(2),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Down, "7"),
                    FixedLengthHexInt::new(2),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Left, "8"),
                    FixedLengthHexInt::new(2),
                )),
                Box::new(Tuple2::new(
                    Dict::new(SlalomBlackCellDir::Right, "9"),
                    FixedLengthHexInt::new(2),
                )),
            ]),
            n_black,
        );
        let seq = sequencer.deserialize_one_elem(ctx, seq_combinator)?;
        let mut idx = 0;
        for y in 0..height {
            for x in 0..width {
                if let SlalomCell::Black(_, _) = grid[y][x] {
                    let (d, n) = seq[idx];
                    grid[y][x] = SlalomCell::Black(d, n);
                    idx += 1;
                }
            }
        }

        Some((sequencer.n_read(), vec![grid]))
    }
}

type PrimitiveProblem = (Vec<Vec<SlalomCell>>, (usize, usize));
type Problem = (Vec<Vec<bool>>, Vec<Gate>, (usize, usize));

pub fn deserialize_problem_as_primitive(url: &str) -> Option<PrimitiveProblem> {
    let combinator = MaybeSkip::new(
        "d/",
        Size::new(Tuple2::new(
            SlalomAuxCombinator,
            MaybeSkip::new("/", DecInt),
        )),
    );
    let (cell, origin) = url_to_problem(combinator, &vec!["slalom"], url)?;
    let width = cell[0].len();

    Some((cell, (origin as usize / width, origin as usize % width)))
}

pub fn parse_primitive_problem(problem: &PrimitiveProblem) -> Problem {
    let (cell, origin) = problem;
    let height = cell.len();
    let width = cell[0].len();

    let mut gates = vec![];
    let mut is_black = vec![vec![false; width]; height];
    for y in 0..height {
        for x in 0..width {
            if let SlalomCell::Black(_, _) = cell[y][x] {
                is_black[y][x] = true;
            }
            if cell[y][x] == SlalomCell::Vertical {
                if y > 0 && cell[y - 1][x] == SlalomCell::Vertical {
                    continue;
                }
                let mut y2 = y;
                while y2 < height && cell[y2][x] == SlalomCell::Vertical {
                    y2 += 1;
                }
                let mut ord = None;
                if y > 0 {
                    if let SlalomCell::Black(SlalomBlackCellDir::Down, n) = cell[y - 1][x] {
                        ord = Some(n);
                    }
                }
                if y2 < height {
                    if let SlalomCell::Black(SlalomBlackCellDir::Up, n) = cell[y2][x] {
                        ord = Some(n);
                    }
                }
                gates.push(Gate {
                    cells: (y..y2).map(|y| (y, x)).collect(),
                    ord,
                });
            } else if cell[y][x] == SlalomCell::Horizontal {
                // horizontal
                if x > 0 && cell[y][x - 1] == SlalomCell::Horizontal {
                    continue;
                }
                let mut x2 = x;
                while x2 < width && cell[y][x2] == SlalomCell::Horizontal {
                    x2 += 1;
                }
                let mut ord = None;
                if x > 0 {
                    if let SlalomCell::Black(SlalomBlackCellDir::Right, n) = cell[y][x - 1] {
                        ord = Some(n);
                    }
                }
                if x2 < width {
                    if let SlalomCell::Black(SlalomBlackCellDir::Left, n) = cell[y][x2] {
                        ord = Some(n);
                    }
                }
                gates.push(Gate {
                    cells: (x..x2).map(|x| (y, x)).collect(),
                    ord,
                });
            }
        }
    }

    (is_black, gates, *origin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slalom_problem() {
        // https://puzsq.jp/main/puzzle_play.php?pid=9522
        let is_black = crate::puzzle::util::tests::to_bool_2d([
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
        ]);

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

        let deserialized = deserialize_problem_as_primitive(
            "https://puzz.link/p?slalom/d/10/10/h133316131f131p1333315131f1333351aj11314333h42g/51",
        );
        assert!(deserialized.is_some());
        let deserialized = parse_primitive_problem(&deserialized.unwrap());
        assert_eq!((is_black, gates, (5, 1)), deserialized);
    }
}
