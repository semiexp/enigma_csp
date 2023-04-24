use super::util;
use crate::graph;
use crate::serializer::strip_prefix;
use crate::solver::{Solver, FALSE};

pub fn solve_ringring(is_black: &[Vec<bool>]) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(is_black);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let horizontal_y = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let horizontal_x = solver.int_var_2d((h, w), 0, (w - 1) as i32);
    let horizontal_h = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let horizontal_w = solver.int_var_2d((h, w), 0, (w - 1) as i32);
    let vertical_y = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let vertical_x = solver.int_var_2d((h, w), 0, (w - 1) as i32);
    let vertical_h = solver.int_var_2d((h, w), 0, (h - 1) as i32);
    let vertical_w = solver.int_var_2d((h, w), 0, (w - 1) as i32);

    for y in 0..h {
        for x in 0..w {
            if is_black[y][x] {
                solver.add_expr(!(is_line.vertex_neighbors((y, x)).any()));
                continue;
            }

            if 0 < y {
                solver.add_expr(is_line.vertical.at((y - 1, x)).imp(
                    vertical_h.at((y - 1, x)).eq(vertical_h.at((y, x)))
                        & vertical_w.at((y - 1, x)).eq(vertical_w.at((y, x)))
                        & vertical_y.at((y - 1, x)).eq(vertical_y.at((y, x)) - 1)
                        & vertical_x.at((y - 1, x)).eq(vertical_x.at((y, x))),
                ));
            }
            if 0 < x {
                solver.add_expr(is_line.horizontal.at((y, x - 1)).imp(
                    horizontal_h.at((y, x - 1)).eq(horizontal_h.at((y, x)))
                        & horizontal_w.at((y, x - 1)).eq(horizontal_w.at((y, x)))
                        & horizontal_y.at((y, x - 1)).eq(horizontal_y.at((y, x)))
                        & horizontal_x.at((y, x - 1)).eq(horizontal_x.at((y, x)) - 1),
                ));
            }
            solver.add_expr(is_line.vertex_neighbors((y, x)).any());

            let is_corner = &solver.bool_var();
            solver.add_expr(is_corner.iff(
                is_line.vertical.at_offset((y, x), (-1, 0), FALSE)
                    ^ is_line.vertical.at_offset((y, x), (0, 0), FALSE),
            ));
            solver.add_expr(is_corner.iff(
                is_line.horizontal.at_offset((y, x), (0, -1), FALSE)
                    ^ is_line.horizontal.at_offset((y, x), (0, 0), FALSE),
            ));
            solver.add_expr(
                (is_corner & !is_line.vertical.at_offset((y, x), (-1, 0), FALSE))
                    .imp(vertical_y.at((y, x)).eq(0)),
            );
            solver.add_expr(
                (is_corner & !is_line.vertical.at_offset((y, x), (0, 0), FALSE))
                    .imp(vertical_y.at((y, x)).eq(vertical_h.at((y, x)))),
            );
            solver.add_expr(
                (is_corner & !is_line.horizontal.at_offset((y, x), (0, -1), FALSE))
                    .imp(horizontal_x.at((y, x)).eq(0)),
            );
            solver.add_expr(
                (is_corner & !is_line.horizontal.at_offset((y, x), (0, 0), FALSE))
                    .imp(horizontal_x.at((y, x)).eq(horizontal_w.at((y, x)))),
            );
            solver.add_expr(is_corner.imp(
                horizontal_y.at((y, x)).eq(vertical_y.at((y, x)))
                    & horizontal_x.at((y, x)).eq(vertical_x.at((y, x)))
                    & horizontal_h.at((y, x)).eq(vertical_h.at((y, x)))
                    & horizontal_w.at((y, x)).eq(vertical_w.at((y, x))),
            ));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<bool>>;

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let serialized = strip_prefix(url)?;
    let pos = serialized.find('/')?;
    let kind = &serialized[0..pos];
    if kind != "ringring" {
        return None;
    }
    let body = &serialized[(pos + 1)..];
    let toks = body.split("/").collect::<Vec<_>>();
    if toks.len() < 3 {
        return None;
    }
    let width = toks[0].parse::<usize>().ok()?;
    let height = toks[1].parse::<usize>().ok()?;
    let mut ret = vec![vec![false; width]; height];
    let body = toks[2].as_bytes();
    let mut pos = 0;
    for &b in body {
        if b == '.' as u8 {
            pos += 36;
        } else if '0' as u8 <= b && b <= '9' as u8 {
            pos += (b - '0' as u8) as usize;
            if pos >= height * width {
                return None;
            }
            ret[pos / width][pos % width] = true;
            pos += 1;
        } else if 'a' as u8 <= b && b <= 'z' as u8 {
            pos += (b - 'a' as u8) as usize + 10;
            if pos >= height * width {
                return None;
            }
            ret[pos / width][pos % width] = true;
            pos += 1;
        }
    }
    Some(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        crate::puzzle::util::tests::to_bool_2d([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
    }

    #[test]
    fn test_ringring_problem() {
        let is_black = problem_for_tests();
        let ans = solve_ringring(&is_black);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 1, 1, 1],
                [1, 1, 0, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 1],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_ringring_deserializer() {
        let url = "https://puzz.link/p?ringring/8/6/063cd4";
        assert_eq!(deserialize_problem(url), Some(problem_for_tests()));
    }
}
