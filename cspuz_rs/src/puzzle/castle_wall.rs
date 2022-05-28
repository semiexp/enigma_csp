use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, FixedLengthHexInt, Grid, HexInt, Map,
    Optionalize, Spaces, Tuple3,
};
use crate::solver::{Solver, FALSE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Unspecified,
    Inside,
    Outside,
}

impl Side {
    fn index(&self) -> i32 {
        match self {
            &Side::Unspecified => 0,
            &Side::Inside => 1,
            &Side::Outside => 2,
        }
    }

    fn from_i32(n: i32) -> Option<Side> {
        match n {
            0 => Some(Side::Unspecified),
            1 => Some(Side::Inside),
            2 => Some(Side::Outside),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrow {
    Unspecified(i32),
    Up(i32),
    Down(i32),
    Left(i32),
    Right(i32),
}

impl Arrow {
    fn dir(&self) -> i32 {
        match self {
            &Arrow::Unspecified(_) => 0,
            &Arrow::Up(_) => 1,
            &Arrow::Down(_) => 2,
            &Arrow::Left(_) => 3,
            &Arrow::Right(_) => 4,
        }
    }

    fn num(&self) -> i32 {
        match self {
            &Arrow::Unspecified(n) => n,
            &Arrow::Up(n) => n,
            &Arrow::Down(n) => n,
            &Arrow::Left(n) => n,
            &Arrow::Right(n) => n,
        }
    }

    fn from_dir_and_num(dir: i32, num: i32) -> Option<Arrow> {
        match dir {
            0 => Some(Arrow::Unspecified(num)),
            1 => Some(Arrow::Up(num)),
            2 => Some(Arrow::Down(num)),
            3 => Some(Arrow::Left(num)),
            4 => Some(Arrow::Right(num)),
            _ => None,
        }
    }
}

pub fn solve_castle_wall(
    clues: &[Vec<Option<(Side, Arrow)>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);
    graph::single_cycle_grid_edges(&mut solver, &is_line);

    let cell_sides = &solver.bool_var_2d((h - 1, w - 1));
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 {
                let a = if x == 0 {
                    FALSE
                } else {
                    cell_sides.at((y, x - 1)).expr()
                };
                let b = if x == w - 1 {
                    FALSE
                } else {
                    cell_sides.at((y, x)).expr()
                };
                solver.add_expr(is_line.vertical.at((y, x)) ^ a.iff(b));
            }
            if x < w - 1 {
                let a = if y == 0 {
                    FALSE
                } else {
                    cell_sides.at((y - 1, x)).expr()
                };
                let b = if y == h - 1 {
                    FALSE
                } else {
                    cell_sides.at((y, x)).expr()
                };
                solver.add_expr(is_line.horizontal.at((y, x)) ^ a.iff(b));
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if let Some((side, arrow)) = clues[y][x] {
                solver.add_expr(!(is_line.vertex_neighbors((y, x)).any()));
                match side {
                    Side::Unspecified => (),
                    Side::Inside => {
                        if y > 0 && x > 0 {
                            solver.add_expr(cell_sides.at((y - 1, x - 1)));
                        } else {
                            return None;
                        }
                    }
                    Side::Outside => {
                        if y > 0 && x > 0 {
                            solver.add_expr(!cell_sides.at((y - 1, x - 1)));
                        }
                    }
                }
                match arrow {
                    Arrow::Unspecified(_) => (),
                    Arrow::Up(n) => {
                        if n >= 0 {
                            solver.add_expr(
                                is_line.vertical.slice_fixed_x((..y, x)).count_true().eq(n),
                            );
                        }
                    }
                    Arrow::Down(n) => {
                        if n >= 0 {
                            solver.add_expr(
                                is_line.vertical.slice_fixed_x((y.., x)).count_true().eq(n),
                            );
                        }
                    }
                    Arrow::Left(n) => {
                        if n >= 0 {
                            solver.add_expr(
                                is_line
                                    .horizontal
                                    .slice_fixed_y((y, ..x))
                                    .count_true()
                                    .eq(n),
                            );
                        }
                    }
                    Arrow::Right(n) => {
                        if n >= 0 {
                            solver.add_expr(
                                is_line
                                    .horizontal
                                    .slice_fixed_y((y, x..))
                                    .count_true()
                                    .eq(n),
                            );
                        }
                    }
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<Option<(Side, Arrow)>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(Map::new(
            Tuple3::new(
                FixedLengthHexInt::new(1),
                FixedLengthHexInt::new(1),
                Choice::new(vec![Box::new(HexInt), Box::new(Dict::new(-1, "."))]),
            ),
            |(side, arrow): (Side, Arrow)| Some((side.index(), arrow.dir(), arrow.num())),
            |(s, d, n)| Some((Side::from_i32(s)?, Arrow::from_dir_and_num(d, n)?)),
        ))),
        Box::new(Spaces::new(None, 'a')),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "castle", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["castle"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        // https://puzsq.jp/main/puzzle_play.php?pid=7711
        let height = 10;
        let width = 10;
        let mut ret = vec![vec![None; width]; height];
        ret[0][0] = Some((Side::Unspecified, Arrow::Down(3)));
        ret[0][3] = Some((Side::Unspecified, Arrow::Down(2)));
        ret[0][6] = Some((Side::Unspecified, Arrow::Down(3)));
        ret[2][9] = Some((Side::Outside, Arrow::Down(4)));
        ret[3][3] = Some((Side::Unspecified, Arrow::Left(2)));
        ret[4][0] = Some((Side::Unspecified, Arrow::Right(4)));
        ret[5][7] = Some((Side::Inside, Arrow::Up(3)));
        ret[6][1] = Some((Side::Unspecified, Arrow::Right(4)));
        ret[6][4] = Some((Side::Unspecified, Arrow::Up(4)));
        ret[8][8] = Some((Side::Outside, Arrow::Up(4)));
        ret[9][1] = Some((Side::Unspecified, Arrow::Up(4)));
        ret[9][4] = Some((Side::Unspecified, Arrow::Up(4)));
        ret
    }

    #[test]
    fn test_castle_wall_problem() {
        let problem = problem_for_tests();
        let ans = solve_castle_wall(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();
        assert_eq!(ans.horizontal[4][7], Some(true));
        assert_eq!(ans.horizontal[4][8], Some(false));
        assert_eq!(ans.vertical[3][8], Some(true));
    }

    #[test]
    fn test_castle_wall_serializer() {
        let problem = problem_for_tests();
        let url =
            "https://puzz.link/p?castle/10/10/023b022b023v224c032f044p113c044b014w214b014b014e";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
