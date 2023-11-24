use super::util;
use crate::graph;
use crate::serializer::{
    from_base16, problem_to_url_with_context, to_base16, url_to_problem, Choice, Combinator,
    Context, ContextBasedGrid, Size, Spaces,
};
use crate::solver::Solver;

pub fn solve_nothree(clues: &[Vec<bool>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);
    let h = (h + 1) / 2;
    let w = (w + 1) / 2;

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    graph::active_vertices_connected_2d(&mut solver, !is_black);
    solver.add_expr(!is_black.conv2d_and((1, 2)));
    solver.add_expr(!is_black.conv2d_and((2, 1)));

    for y in 0..(h * 2 - 1) {
        for x in 0..(w * 2 - 1) {
            if clues[y][x] {
                solver.add_expr(
                    is_black
                        .slice(((y / 2)..=((y + 1) / 2), (x / 2)..=((x + 1) / 2)))
                        .count_true()
                        .eq(1),
                );
            }
        }
    }
    for y in 0..h {
        for x1 in 0..w {
            for x2 in (x1 + 1)..w {
                let x3 = x2 * 2 - x1;
                if x3 >= w {
                    continue;
                }
                solver.add_expr(
                    !is_black.at((y, x1))
                        | !is_black.at((y, x2))
                        | !is_black.at((y, x3))
                        | is_black.slice_fixed_y((y, (x1 + 1)..x2)).any()
                        | is_black.slice_fixed_y((y, (x2 + 1)..x3)).any(),
                );
            }
        }
    }
    for x in 0..w {
        for y1 in 0..h {
            for y2 in (y1 + 1)..h {
                let y3 = y2 * 2 - y1;
                if y3 >= h {
                    continue;
                }
                solver.add_expr(
                    !is_black.at((y1, x))
                        | !is_black.at((y2, x))
                        | !is_black.at((y3, x))
                        | is_black.slice_fixed_x(((y1 + 1)..y2, x)).any()
                        | is_black.slice_fixed_x(((y2 + 1)..y3, x)).any(),
                );
            }
        }
    }
    solver.irrefutable_facts().map(|f| f.get(is_black))
}

type Problem = Vec<Vec<bool>>;

pub struct SizeDoubler<S> {
    base_serializer: S,
}

impl<S> SizeDoubler<S> {
    fn new(base_serializer: S) -> SizeDoubler<S> {
        SizeDoubler { base_serializer }
    }
}

impl<S, T> Combinator<T> for SizeDoubler<S>
where
    S: Combinator<T>,
{
    fn serialize(&self, ctx: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
        let ctx = Context {
            height: ctx.height.map(|a| a * 2 - 1),
            width: ctx.width.map(|a| a * 2 - 1),
            ..*ctx
        };
        self.base_serializer.serialize(&ctx, input)
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        let ctx = Context {
            height: ctx.height.map(|a| a * 2 - 1),
            width: ctx.width.map(|a| a * 2 - 1),
            ..*ctx
        };
        self.base_serializer.deserialize(&ctx, input)
    }
}

pub struct NothreeClueCombinator;

impl Combinator<bool> for NothreeClueCombinator {
    fn serialize(&self, _: &Context, input: &[bool]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        if !input[0] {
            return None;
        }
        let mut n_spaces = 0;
        while n_spaces < 7 && 1 + n_spaces < input.len() && !input[1 + n_spaces] {
            n_spaces += 1;
        }
        Some((1 + n_spaces, vec![to_base16(n_spaces as i32 * 2)]))
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<bool>)> {
        if input.len() == 0 {
            return None;
        }
        let c = from_base16(input[0])?;
        if c % 2 != 0 {
            return None;
        }
        let spaces = c / 2;
        let mut ret = vec![true];
        for _ in 0..spaces {
            ret.push(false);
        }
        Some((1, ret))
    }
}

fn combinator() -> impl Combinator<Problem> {
    Size::new(SizeDoubler::new(ContextBasedGrid::new(Choice::new(vec![
        Box::new(NothreeClueCombinator),
        Box::new(Spaces::new(false, 'g')),
    ]))))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(problem);
    problem_to_url_with_context(
        combinator(),
        "nothree",
        problem.clone(),
        &Context::sized((h + 1) / 2, (w + 1) / 2),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["nothree"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        crate::puzzle::util::tests::to_bool_2d([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
    }

    #[test]
    fn test_nothree_problem() {
        let problem = problem_for_tests();
        let ans = solve_nothree(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_firefly_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?nothree/6/5/ger26eneq22eleq";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
