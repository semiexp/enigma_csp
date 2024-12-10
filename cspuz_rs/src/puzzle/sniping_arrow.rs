use super::util;
use crate::graph;
use crate::items::Arrow;
use crate::serializer::{
    get_kudamono_url_info, kudamono_url_info_to_problem, problem_to_kudamono_url_grid, Choice,
    Combinator, Context, DecInt, Dict, KudamonoGrid, Optionalize, PrefixAndSuffix, Sequencer,
};
use crate::solver::{int_constant, Solver};

pub fn solve_sniping_arrow(
    clues: &[Vec<Option<(Option<i32>, Option<Arrow>)>>],
) -> Option<(
    graph::BoolGridEdgesIrrefutableFacts,
    Vec<Vec<Option<Arrow>>>,
)> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();

    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    // 1: up, 2: down, 3: left, 4: right
    let arrow_head = &solver.int_var_2d((h, w), 0, 4);
    solver.add_answer_key_int(arrow_head);

    solver.add_expr(!(arrow_head.slice((.., ..(w - 1))).ne(0) & arrow_head.slice((.., 1..)).ne(0)));
    solver.add_expr(!(arrow_head.slice((..(h - 1), ..)).ne(0) & arrow_head.slice((1.., ..)).ne(0)));

    // From the rule, it follows that, if a cell points to another cell with the same direction,
    // these two cells must belong to the same "arrow".
    let direction = &solver.int_var_2d((h, w), 0, 4);
    let num = &solver.int_var_2d((h, w), 0, h.max(w) as i32);

    for y in 0..h {
        for x in 0..w {
            if let Some((n, ar)) = clues[y][x] {
                for (d, offset) in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))] {
                    solver.add_expr(arrow_head.at((y, x)).eq(d).iff(
                        direction.at((y, x)).eq(d)
                            & direction.at_offset((y, x), offset, int_constant(0)).ne(d),
                    ));
                }

                solver.add_expr(direction.at((y, x)).ne(0));
                match ar {
                    None => (),
                    Some(Arrow::Up) => solver.add_expr(arrow_head.at((y, x)).eq(1)),
                    Some(Arrow::Down) => solver.add_expr(arrow_head.at((y, x)).eq(2)),
                    Some(Arrow::Left) => solver.add_expr(arrow_head.at((y, x)).eq(3)),
                    Some(Arrow::Right) => solver.add_expr(arrow_head.at((y, x)).eq(4)),
                    Some(Arrow::Unspecified) => panic!(),
                }

                if let Some(n) = n {
                    solver.add_expr(num.at((y, x)).eq(n));

                    if y > 0 {
                        solver.add_expr(
                            direction
                                .at((y, x))
                                .eq(2)
                                .imp(direction.at((y - 1, x)).ne(2)),
                        );
                    }
                    if y < h - 1 {
                        solver.add_expr(
                            direction
                                .at((y, x))
                                .eq(1)
                                .imp(direction.at((y + 1, x)).ne(1)),
                        );
                    }
                    if x > 0 {
                        solver.add_expr(
                            direction
                                .at((y, x))
                                .eq(4)
                                .imp(direction.at((y, x - 1)).ne(4)),
                        );
                    }
                    if x < w - 1 {
                        solver.add_expr(
                            direction
                                .at((y, x))
                                .eq(3)
                                .imp(direction.at((y, x + 1)).ne(3)),
                        );
                    }
                }
            } else {
                solver.add_expr(arrow_head.at((y, x)).eq(0));
                solver.add_expr(direction.at((y, x)).eq(0));
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if clues[y][x].is_none() {
                solver.add_expr(num.at((y, x)).eq(0));
            } else {
                solver.add_expr(num.at((y, x)).ge(2));
                solver.add_expr(
                    direction.at((y, x)).eq(1).imp(
                        num.at((y, x)).eq(direction
                            .slice_fixed_x((..y, x))
                            .eq(1)
                            .reverse()
                            .consecutive_prefix_true()
                            + direction
                                .slice_fixed_x((y.., x))
                                .eq(1)
                                .consecutive_prefix_true()),
                    ),
                );
                solver.add_expr(
                    direction.at((y, x)).eq(2).imp(
                        num.at((y, x)).eq(direction
                            .slice_fixed_x((..y, x))
                            .eq(2)
                            .reverse()
                            .consecutive_prefix_true()
                            + direction
                                .slice_fixed_x((y.., x))
                                .eq(2)
                                .consecutive_prefix_true()),
                    ),
                );
                solver.add_expr(
                    direction.at((y, x)).eq(3).imp(
                        num.at((y, x)).eq(direction
                            .slice_fixed_y((y, ..x))
                            .eq(3)
                            .reverse()
                            .consecutive_prefix_true()
                            + direction
                                .slice_fixed_y((y, x..))
                                .eq(3)
                                .consecutive_prefix_true()),
                    ),
                );
                solver.add_expr(
                    direction.at((y, x)).eq(4).imp(
                        num.at((y, x)).eq(direction
                            .slice_fixed_y((y, ..x))
                            .eq(4)
                            .reverse()
                            .consecutive_prefix_true()
                            + direction
                                .slice_fixed_y((y, x..))
                                .eq(4)
                                .consecutive_prefix_true()),
                    ),
                );
            }
        }
    }
    for y in 0..h {
        for x in 0..(w - 1) {
            if clues[y][x].is_none() || clues[y][x + 1].is_none() {
                solver.add_expr(!is_line.horizontal.at((y, x)));
            } else {
                let a = direction.at((y, x));
                let b = direction.at((y, x + 1));
                solver.add_expr(
                    is_line
                        .horizontal
                        .at((y, x))
                        .iff(a.eq(b) & (a.eq(3) | a.eq(4))),
                );
                solver.add_expr(arrow_head.at((y, x)).eq(0) | arrow_head.at((y, x + 1)).eq(0));
                solver.add_expr(
                    (!is_line.horizontal.at((y, x))).imp(num.at((y, x)).ne(num.at((y, x + 1)))),
                );
            }
        }
    }

    for y in 0..(h - 1) {
        for x in 0..w {
            if clues[y][x].is_none() || clues[y + 1][x].is_none() {
                solver.add_expr(!is_line.vertical.at((y, x)));
            } else {
                let a = direction.at((y, x));
                let b = direction.at((y + 1, x));
                solver.add_expr(
                    is_line
                        .vertical
                        .at((y, x))
                        .iff(a.eq(b) & (a.eq(1) | a.eq(2))),
                );
                solver.add_expr(arrow_head.at((y, x)).eq(0) | arrow_head.at((y + 1, x)).eq(0));
                solver.add_expr(
                    (!is_line.vertical.at((y, x))).imp(num.at((y, x)).ne(num.at((y + 1, x)))),
                );
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if clues[y][x].is_none() {
                continue;
            }

            for (d, (dy, dx)) in [(1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))] {
                let mut y2 = y as i32 + dy;
                let mut x2 = x as i32 + dx;

                while 0 <= y2
                    && y2 < h as i32
                    && 0 <= x2
                    && x2 < w as i32
                    && clues[y2 as usize][x2 as usize].is_some()
                {
                    solver.add_expr(
                        arrow_head
                            .at((y, x))
                            .eq(d)
                            .imp(arrow_head.at((y2 as usize, x2 as usize)).eq(0)),
                    );
                    y2 += dy;
                    x2 += dx;
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| {
        let arrow = f
            .get(arrow_head)
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|v| match v {
                        Some(0) => Some(Arrow::Unspecified),
                        Some(1) => Some(Arrow::Up),
                        Some(2) => Some(Arrow::Down),
                        Some(3) => Some(Arrow::Left),
                        Some(4) => Some(Arrow::Right),
                        None => None,
                        _ => panic!(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        (f.get(is_line), arrow)
    })
}

struct SnipingArrowClueCombinator;

impl Combinator<(Option<i32>, Option<Arrow>)> for SnipingArrowClueCombinator {
    fn serialize(
        &self,
        ctx: &Context,
        input: &[(Option<i32>, Option<Arrow>)],
    ) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let (n, ar) = input[0];

        let mut ret = vec![];
        if let Some(n) = n {
            ret.push('(' as u8);
            ret.extend(DecInt.serialize(ctx, &[n])?.1);
            ret.push(')' as u8);
        }
        if let Some(ar) = ar {
            match ar {
                Arrow::Unspecified => return None,
                Arrow::Up => ret.push('u' as u8),
                Arrow::Down => ret.push('d' as u8),
                Arrow::Left => ret.push('l' as u8),
                Arrow::Right => ret.push('r' as u8),
            }
        }

        Some((1, ret))
    }

    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<(Option<i32>, Option<Arrow>)>)> {
        let mut sequencer = Sequencer::new(input);

        let n = sequencer
            .deserialize(ctx, PrefixAndSuffix::new("(", DecInt, ")"))
            .map(|x| x[0]);
        let ar = sequencer
            .deserialize(
                ctx,
                Choice::new(vec![
                    Box::new(Dict::new(Arrow::Up, "u")),
                    Box::new(Dict::new(Arrow::Down, "d")),
                    Box::new(Dict::new(Arrow::Left, "l")),
                    Box::new(Dict::new(Arrow::Right, "r")),
                ]),
            )
            .map(|x| x[0]);

        if n.is_none() && ar.is_none() {
            return None;
        }

        Some((sequencer.n_read(), vec![(n, ar)]))
    }
}

type Problem = Vec<Vec<Option<(Option<i32>, Option<Arrow>)>>>;

fn combinator() -> impl Combinator<Problem> {
    KudamonoGrid::new(
        Choice::new(vec![
            Box::new(Dict::new(None, "z")),
            Box::new(Optionalize::new(SnipingArrowClueCombinator)),
        ]),
        Some((None, None)),
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_kudamono_url_grid(combinator(), "sniping-arrow", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let info = get_kudamono_url_info(url)?;
    kudamono_url_info_to_problem(combinator(), info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        let mut ret = vec![vec![Some((None, None)); 6]; 5];
        ret[0][0] = None;
        ret[0][1] = Some((None, Some(Arrow::Up)));
        ret[2][1] = Some((Some(3), None));
        ret[2][3] = None;
        ret[3][5] = None;
        ret[4][0] = None;
        ret[4][5] = None;

        ret
    }

    #[test]
    fn test_sniping_arrow_problem() {
        let clues = problem_for_tests();
        let ans = solve_sniping_arrow(&clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_edges = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
        };

        let mut expected_arrows = vec![vec![Some(Arrow::Unspecified); 6]; 5];
        expected_arrows[0][1] = Some(Arrow::Up);
        expected_arrows[0][5] = Some(Arrow::Right);
        expected_arrows[1][3] = Some(Arrow::Left);
        expected_arrows[2][0] = Some(Arrow::Down);
        expected_arrows[2][2] = Some(Arrow::Down);
        expected_arrows[2][5] = Some(Arrow::Right);
        expected_arrows[3][4] = Some(Arrow::Right);
        expected_arrows[4][1] = Some(Arrow::Left);

        assert_eq!(ans, (expected_edges, expected_arrows));
    }

    #[test]
    fn test_sniping_arrow_serializer() {
        let problem = problem_for_tests();
        let url =
            "https://pedros.works/paper-puzzle-player?W=6x5&L=z0z4(3)3u2z8z8z1&G=sniping-arrow";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
