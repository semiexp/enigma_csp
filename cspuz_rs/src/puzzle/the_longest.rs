use crate::graph;
use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, Combinator, Context,
};
use crate::solver::{int_constant, BoolExpr, BoolVarArray1D, IntExpr, Solver, FALSE, TRUE};

pub fn solve_the_longest(
    clues: &graph::GridEdges<Vec<Vec<bool>>>,
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let h = clues.vertical.len();
    let w = clues.vertical[0].len() - 1;

    let mut solver = Solver::new();
    let ans = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&ans.horizontal);
    solver.add_answer_key_bool(&ans.vertical);

    let ans_outer = &graph::BoolGridEdges::new(&mut solver, (h, w));
    solver.add_expr(ans_outer.horizontal.slice_fixed_y((0, ..)));
    solver.add_expr(ans_outer.horizontal.slice_fixed_y((h, ..)));
    solver.add_expr(ans_outer.vertical.slice_fixed_x((.., 0)));
    solver.add_expr(ans_outer.vertical.slice_fixed_x((.., w)));
    solver.add_expr(ans_outer.horizontal.slice((1..h, ..)).iff(&ans.horizontal));
    solver.add_expr(ans_outer.vertical.slice((.., 1..w)).iff(&ans.vertical));

    fn compute_edge_len(edges: BoolVarArray1D, walls: BoolVarArray1D) -> IntExpr {
        assert_eq!(edges.len(), walls.len());

        let mut ret = int_constant(0);

        for i in 0..edges.len() {
            let i = edges.len() - 1 - i;
            ret = (edges.at(i) & !walls.at(i)).ite(ret + 1, 0);
        }

        ret
    }

    let edge_len_up = &solver.int_var_2d((h, w), 0, w as i32);
    let edge_len_down = &solver.int_var_2d((h, w), 0, w as i32);
    let edge_len_left = &solver.int_var_2d((h, w), 0, h as i32);
    let edge_len_right = &solver.int_var_2d((h, w), 0, h as i32);

    for y in 0..h {
        for x in 0..w {
            solver.add_expr(
                edge_len_up
                    .at((y, x))
                    .eq(ans_outer.horizontal.at((y, x)).ite(
                        compute_edge_len(
                            ans_outer.horizontal.slice_fixed_y((y, (x + 1)..)),
                            ans.vertical.slice_fixed_y((y, x..)),
                        ) + compute_edge_len(
                            ans_outer.horizontal.slice_fixed_y((y, ..x)).reverse(),
                            ans.vertical.slice_fixed_y((y, ..x)).reverse(),
                        ) + 1,
                        0,
                    )),
            );
            solver.add_expr(
                edge_len_down
                    .at((y, x))
                    .eq(ans_outer.horizontal.at((y + 1, x)).ite(
                        compute_edge_len(
                            ans_outer.horizontal.slice_fixed_y((y + 1, (x + 1)..)),
                            ans.vertical.slice_fixed_y((y, x..)),
                        ) + compute_edge_len(
                            ans_outer.horizontal.slice_fixed_y((y + 1, ..x)).reverse(),
                            ans.vertical.slice_fixed_y((y, ..x)).reverse(),
                        ) + 1,
                        0,
                    )),
            );
            solver.add_expr(
                edge_len_left
                    .at((y, x))
                    .eq(ans_outer.vertical.at((y, x)).ite(
                        compute_edge_len(
                            ans_outer.vertical.slice_fixed_x(((y + 1).., x)),
                            ans.horizontal.slice_fixed_x((y.., x)),
                        ) + compute_edge_len(
                            ans_outer.vertical.slice_fixed_x((..y, x)).reverse(),
                            ans.horizontal.slice_fixed_x((..y, x)).reverse(),
                        ) + 1,
                        0,
                    )),
            );
            solver.add_expr(
                edge_len_right
                    .at((y, x))
                    .eq(ans_outer.vertical.at((y, x + 1)).ite(
                        compute_edge_len(
                            ans_outer.vertical.slice_fixed_x(((y + 1).., x + 1)),
                            ans.horizontal.slice_fixed_x((y.., x)),
                        ) + compute_edge_len(
                            ans_outer.vertical.slice_fixed_x((..y, x + 1)).reverse(),
                            ans.horizontal.slice_fixed_x((..y, x)).reverse(),
                        ) + 1,
                        0,
                    )),
            );
        }
    }

    let edge_max = &solver.int_var_2d((h, w), 0, (w.max(h)) as i32);
    solver.add_expr(edge_len_up.le(edge_max));
    solver.add_expr(edge_len_down.le(edge_max));
    solver.add_expr(edge_len_left.le(edge_max));
    solver.add_expr(edge_len_right.le(edge_max));

    solver.add_expr(
        (!&ans.horizontal).imp(
            edge_max
                .slice((..(h - 1), ..))
                .eq(edge_max.slice((1.., ..))),
        ),
    );
    solver.add_expr(
        (!&ans.vertical).imp(
            edge_max
                .slice((.., ..(w - 1)))
                .eq(edge_max.slice((.., 1..))),
        ),
    );

    let mut aux_graph = graph::Graph::new(h * w + 1);
    let mut aux_edges = vec![];

    let mut add_edge = |u: usize, v: usize, val: BoolExpr| {
        aux_graph.add_edge(u, v);
        aux_edges.push(val);
    };

    let terminal = h * w;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;

            add_edge(
                i,
                terminal,
                ans_outer.horizontal.at((y, x)) & edge_len_up.at((y, x)).eq(edge_max.at((y, x)))
                    | ans_outer.horizontal.at((y + 1, x))
                        & edge_len_down.at((y, x)).eq(edge_max.at((y, x)))
                    | ans_outer.vertical.at((y, x))
                        & edge_len_left.at((y, x)).eq(edge_max.at((y, x)))
                    | ans_outer.vertical.at((y, x + 1))
                        & edge_len_right.at((y, x)).eq(edge_max.at((y, x))),
            );

            if x < w - 1 {
                add_edge(i, i + 1, !ans.vertical.at((y, x)).expr());
            }
            if y < h - 1 {
                add_edge(i, i + w, !ans.horizontal.at((y, x)).expr());
            }
        }
    }

    graph::active_vertices_connected_via_active_edges(
        &mut solver,
        &vec![TRUE; h * w + 1],
        &aux_edges,
        &aux_graph,
    );

    for y in 0..=h {
        for x in 0..w {
            let up;
            if y == 0 {
                up = FALSE;
            } else {
                up = edge_len_down.at((y - 1, x)).eq(edge_max.at((y - 1, x)));
            }

            let down;
            if y == h {
                down = FALSE;
            } else {
                down = edge_len_up.at((y, x)).eq(edge_max.at((y, x)));
            }

            if clues.horizontal[y][x] {
                solver.add_expr(up | down);
                solver.add_expr(ans_outer.horizontal.at((y, x)));
            } else {
                solver.add_expr(ans_outer.horizontal.at((y, x)).imp(!(up | down)));
            }
        }
    }
    for y in 0..h {
        for x in 0..=w {
            let left;
            if x == 0 {
                left = FALSE;
            } else {
                left = edge_len_right.at((y, x - 1)).eq(edge_max.at((y, x - 1)));
            }

            let right;
            if x == w {
                right = FALSE;
            } else {
                right = edge_len_left.at((y, x)).eq(edge_max.at((y, x)));
            }

            if clues.vertical[y][x] {
                solver.add_expr(left | right);
                solver.add_expr(ans_outer.vertical.at((y, x)));
            } else {
                solver.add_expr(ans_outer.vertical.at((y, x)).imp(!(left | right)));
            }
        }
    }

    // ensure that there are no extra edges
    let (edges, g) = ans.clone().dual().representation();
    let g2 = (0..g.n_edges()).map(|i| g[i]).collect::<Vec<_>>();
    solver.add_graph_division(&vec![None; h * w], &g2, &edges);

    solver.irrefutable_facts().map(|f| f.get(ans))
}

type Problem = graph::GridEdges<Vec<Vec<bool>>>;

pub struct KudamonoInnerBorder;

impl Combinator<graph::GridEdges<Vec<Vec<bool>>>> for KudamonoInnerBorder {
    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<graph::GridEdges<Vec<Vec<bool>>>>)> {
        // TODO: consumes the entire input
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let mut border = graph::GridEdges {
            vertical: vec![vec![false; width + 1]; height],
            horizontal: vec![vec![false; width]; height + 1],
        };

        let mut idx = 0;
        let mut pos = 0;

        while idx < input.len() {
            if '0' as u8 <= input[idx] && input[idx] <= '9' as u8 {
                let mut num_end = idx;
                let mut n = 0;
                while num_end < input.len()
                    && '0' as u8 <= input[num_end]
                    && input[num_end] <= '9' as u8
                {
                    n *= 10;
                    n += (input[num_end] - '0' as u8) as usize;
                    num_end += 1;
                }
                pos += n;
                idx = num_end;
            } else {
                let mut y = height - pos % (height + 1);
                let mut x = pos / (height + 1);

                if input[idx] != 'R' as u8
                    && input[idx] != 'L' as u8
                    && input[idx] != 'U' as u8
                    && input[idx] != 'D' as u8
                {
                    return None;
                }

                while idx < input.len() {
                    if input[idx] == 'L' as u8 {
                        if x == 0 {
                            return None;
                        }
                        border.horizontal[y][x - 1] = true;
                        x -= 1;
                    } else if input[idx] == 'R' as u8 {
                        if x >= width {
                            return None;
                        }
                        border.horizontal[y][x] = true;
                        x += 1;
                    } else if input[idx] == 'U' as u8 {
                        if y == 0 {
                            return None;
                        }
                        border.vertical[y - 1][x] = true;
                        y -= 1;
                    } else if input[idx] == 'D' as u8 {
                        if y >= height {
                            return None;
                        }
                        border.vertical[y][x] = true;
                        y += 1;
                    } else {
                        break;
                    }
                    idx += 1;
                }
            }
        }

        Some((idx, vec![border]))
    }

    fn serialize(
        &self,
        _ctx: &Context,
        _input: &[graph::GridEdges<Vec<Vec<bool>>>],
    ) -> Option<(usize, Vec<u8>)> {
        unimplemented!();
    }
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W")?)?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    KudamonoInnerBorder
        .deserialize(&ctx, parsed.get("SIE")?.as_bytes())?
        .1
        .pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        graph::GridEdges {
            horizontal: crate::puzzle::util::tests::to_bool_2d([
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_bool_2d([
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ]),
        }
    }

    #[test]
    fn test_the_longest_problem() {
        let clues = problem_for_tests();
        let ans = solve_the_longest(&clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 0],
            ]),
        };

        assert_eq!(ans, expected);
    }

    #[test]
    fn test_cbpl_serializer() {
        let problem = problem_for_tests();
        let url =
            "https://pedros.works/paper-puzzle-player?W=5x4&SIE=0RRR2UU7RRR2UU15UUU&G=the-longest";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
