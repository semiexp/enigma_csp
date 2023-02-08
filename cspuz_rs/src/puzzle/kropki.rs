use crate::graph::InnerGridEdges;
use crate::serializer::{
    map_2d, problem_to_url_with_context, url_to_problem, Combinator, Context, ContextBasedGrid,
    MultiDigit, Sequencer, Size,
};
use crate::solver::{IntVar, Solver};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KropkiClue {
    None,
    White,
    Black,
}

fn add_kropki_constraint(solver: &mut Solver, a: &IntVar, b: &IntVar, clue: KropkiClue) {
    match clue {
        KropkiClue::None => {
            solver.add_expr(a.ne(b + 1));
            solver.add_expr(a.ne(b - 1));
            solver.add_expr(a.ne(b + b));
            solver.add_expr(b.ne(a + a));
        }
        KropkiClue::White => solver.add_expr(a.eq(b + 1) | a.eq(b - 1)),
        KropkiClue::Black => solver.add_expr(a.eq(b + b) | b.eq(a + a)),
    }
}

pub fn solve_kropki(clues: &InnerGridEdges<Vec<Vec<KropkiClue>>>) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = clues.base_shape();
    assert_eq!(h, w);
    let n = h;

    let mut solver = Solver::new();
    let num = &solver.int_var_2d((n, n), 1, n as i32);
    solver.add_answer_key_int(num);

    for i in 0..n {
        solver.all_different(num.slice_fixed_y((i, ..)));
        solver.all_different(num.slice_fixed_x((.., i)));
    }

    for y in 0..n {
        for x in 0..n {
            if y < n - 1 {
                add_kropki_constraint(
                    &mut solver,
                    &num.at((y, x)),
                    &num.at((y + 1, x)),
                    clues.horizontal[y][x],
                );
            }
            if x < n - 1 {
                add_kropki_constraint(
                    &mut solver,
                    &num.at((y, x)),
                    &num.at((y, x + 1)),
                    clues.vertical[y][x],
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(num))
}

fn kropi_clue_to_i32(clue: &KropkiClue) -> i32 {
    match *clue {
        KropkiClue::None => 0,
        KropkiClue::White => 1,
        KropkiClue::Black => 2,
    }
}

fn i32_to_kropki_clue(n: &i32) -> KropkiClue {
    match *n {
        0 => KropkiClue::None,
        1 => KropkiClue::White,
        2 => KropkiClue::Black,
        _ => panic!(),
    }
}

pub struct KropkiCombinator;

impl Combinator<InnerGridEdges<Vec<Vec<KropkiClue>>>> for KropkiCombinator {
    fn serialize(
        &self,
        ctx: &Context,
        input: &[InnerGridEdges<Vec<Vec<KropkiClue>>>],
    ) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let vertical_i32 = map_2d(&input[0].vertical, kropi_clue_to_i32);
        let horizontal_i32 = map_2d(&input[0].horizontal, kropi_clue_to_i32);

        let ctx_grid = ContextBasedGrid::new(MultiDigit::new(3, 3));
        let mut ret = vec![];
        let (_, app) = ctx_grid.serialize(
            &Context {
                height: Some(height),
                width: Some(width - 1),
                ..*ctx
            },
            &vec![vertical_i32],
        )?;
        ret.extend(app);
        let (_, app) = ctx_grid.serialize(
            &Context {
                height: Some(height - 1),
                width: Some(width),
                ..*ctx
            },
            &vec![horizontal_i32],
        )?;
        ret.extend(app);

        Some((1, ret))
    }

    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<InnerGridEdges<Vec<Vec<KropkiClue>>>>)> {
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();
        let mut sequencer = Sequencer::new(input);
        let ctx_grid = ContextBasedGrid::new(MultiDigit::new(3, 3));

        let vertical_i32 = sequencer.deserialize_one_elem(
            &Context {
                height: Some(height),
                width: Some(width - 1),
                ..*ctx
            },
            &ctx_grid,
        )?;
        let horizontal_i32 = sequencer.deserialize_one_elem(
            &Context {
                height: Some(height - 1),
                width: Some(width),
                ..*ctx
            },
            &ctx_grid,
        )?;

        let vertical = map_2d(&vertical_i32, i32_to_kropki_clue);
        let horizontal = map_2d(&horizontal_i32, i32_to_kropki_clue);

        Some((
            sequencer.n_read(),
            vec![InnerGridEdges {
                vertical,
                horizontal,
            }],
        ))
    }
}

type Problem = InnerGridEdges<Vec<Vec<KropkiClue>>>;

fn combinator() -> impl Combinator<Problem> {
    Size::new(KropkiCombinator)
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (height, width) = problem.base_shape();
    problem_to_url_with_context(
        combinator(),
        "kropki",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["kropki"], url)
}

#[cfg(test)]
mod tests {
    use super::super::util;
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        InnerGridEdges {
            horizontal: vec![
                vec![KropkiClue::None, KropkiClue::White, KropkiClue::White, KropkiClue::White],
                vec![KropkiClue::None, KropkiClue::None, KropkiClue::Black, KropkiClue::Black],
                vec![KropkiClue::White, KropkiClue::None, KropkiClue::White, KropkiClue::White],
            ],
            vertical: vec![
                vec![KropkiClue::Black, KropkiClue::Black, KropkiClue::None],
                vec![KropkiClue::None, KropkiClue::White, KropkiClue::Black],
                vec![KropkiClue::None, KropkiClue::None, KropkiClue::Black],
                vec![KropkiClue::Black, KropkiClue::White, KropkiClue::None],
            ],
        }
    }

    #[test]
    fn test_kropki_problem() {
        let problem = problem_for_tests();
        let ans = solve_kropki(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_2d([
            [4, 2, 1, 3],
            [1, 3, 2, 4],
            [3, 1, 4, 2],
            [2, 4, 3, 1],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_kropki_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?kropki/4/4/o52l49p4";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
