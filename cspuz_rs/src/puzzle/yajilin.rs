use crate::graph;
use crate::serializer::{
    from_base16, is_hex, problem_to_url, to_base16, url_to_problem, Choice, Combinator, Context,
    Grid, MaybeSkip, Optionalize, Spaces,
};
use crate::solver::Solver;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
                    YajilinClue::Unspecified(n) => {
                        if n >= 0 {
                            unimplemented!();
                        }
                    }
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

struct YajilinClueCombinator;

impl Combinator<YajilinClue> for YajilinClueCombinator {
    fn serialize(&self, _: &Context, input: &[YajilinClue]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let (dir, n) = match input[0] {
            YajilinClue::Unspecified(n) => (0, n),
            YajilinClue::Up(n) => (1, n),
            YajilinClue::Down(n) => (2, n),
            YajilinClue::Left(n) => (3, n),
            YajilinClue::Right(n) => (4, n),
        };
        if n == -1 {
            Some((1, vec![dir + ('0' as u8), '.' as u8]))
        } else if 0 <= n && n < 16 {
            Some((1, vec![dir + ('0' as u8), to_base16(n)]))
        } else if 16 <= n && n < 256 {
            Some((
                1,
                vec![dir + ('5' as u8), to_base16(n >> 4), to_base16(n & 15)],
            ))
        } else {
            None
        }
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<YajilinClue>)> {
        if input.len() < 2 {
            return None;
        }
        let dir = input[0];
        if !('0' as u8 <= dir && dir <= '9' as u8) {
            return None;
        }
        let dir = dir - '0' as u8;
        let n;
        let n_read;
        {
            if dir < 5 {
                if input[1] == '.' as u8 {
                    n = -1;
                } else {
                    if !is_hex(input[1]) {
                        return None;
                    }
                    n = from_base16(input[1]);
                }
                n_read = 2;
            } else {
                if input.len() < 3 || !is_hex(input[1]) || !is_hex(input[2]) {
                    return None;
                }
                n = (from_base16(input[1]) << 4) | from_base16(input[2]);
                n_read = 3;
            }
        }
        Some((
            n_read,
            vec![match dir % 5 {
                0 => YajilinClue::Unspecified(n),
                1 => YajilinClue::Up(n),
                2 => YajilinClue::Down(n),
                3 => YajilinClue::Left(n),
                4 => YajilinClue::Right(n),
                _ => unreachable!(),
            }],
        ))
    }
}

type Problem = Vec<Vec<Option<YajilinClue>>>;

fn combinator() -> impl Combinator<Problem> {
    MaybeSkip::new(
        "b/",
        Grid::new(Choice::new(vec![
            Box::new(Optionalize::new(YajilinClueCombinator)),
            Box::new(Spaces::new(None, 'a')),
        ])),
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "yajilin", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["yajilin", "yajirin"], url)
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

        assert_eq!(
            serialize_problem(&problem),
            Some(String::from(
                "https://puzz.link/p?yajilin/10/10/w32a41b21a21l22e30m21a12b11r20d30g"
            ))
        );
        assert_eq!(
            deserialize_problem(
                "https://puzz.link/p?yajilin/10/10/w32a41b21a21l22e30m21a12b11r20d30g"
            ),
            Some(problem.clone())
        );

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

    #[test]
    fn test_yajilin_clue_combinator() {
        let ctx = &Context::new();
        let combinator = YajilinClueCombinator;

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[YajilinClue::Up(0)]),
            Some((1, Vec::from("10")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[YajilinClue::Down(3)]),
            Some((1, Vec::from("23")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[YajilinClue::Left(-1)]),
            Some((1, Vec::from("3.")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[YajilinClue::Right(63)]),
            Some((1, Vec::from("93f")))
        );

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "105".as_bytes()),
            Some((2, vec![YajilinClue::Up(0)]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "23".as_bytes()),
            Some((2, vec![YajilinClue::Down(3)]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "3.".as_bytes()),
            Some((2, vec![YajilinClue::Left(-1)]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "93f".as_bytes()),
            Some((3, vec![YajilinClue::Right(63)]))
        );
    }
}
