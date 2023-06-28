use super::util;
use crate::graph;
use crate::serializer::{get_kudamono_url_info, lexicographic_order, Context, DecInt, Sequencer};
use crate::solver::{any, int_constant, Solver, TRUE};

pub fn solve_crosswall(
    clues: &[Vec<Option<(i32, i32)>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    graph::crossable_single_cycle_grid_edges(&mut solver, &is_line);

    let mut sizes = vec![];
    let mut edges = vec![];
    let mut edge_vars = vec![];
    for y in 0..h {
        for x in 0..w {
            let n = match clues[y][x] {
                Some((n, _)) if n > 0 => n,
                _ => -1,
            };
            if n >= 0 {
                sizes.push(Some(int_constant(n)));
            } else {
                sizes.push(None);
            }
            if y < h - 1 {
                edges.push((y * w + x, (y + 1) * w + x));
                edge_vars.push(is_line.horizontal.at((y + 1, x)));
            }
            if x < w - 1 {
                edges.push((y * w + x, y * w + x + 1));
                edge_vars.push(is_line.vertical.at((y, x + 1)));
            }
        }
    }
    solver.add_graph_division(&sizes, &edges, &edge_vars);

    let mut aux_graph = graph::Graph::new(2 * h * w + 1);
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 {
                aux_graph.add_edge(y * w + x, (y + 1) * w + x);
            }
            if x < w - 1 {
                aux_graph.add_edge(y * w + x, y * w + x + 1);
            }
            aux_graph.add_edge(y * w + x, (y + h) * w + x);
            aux_graph.add_edge((y + h) * w + x, 2 * h * w);
        }
    }

    let max_level = (h.min(w) + 1) as i32 / 2;
    let levels = &solver.int_var_2d((h, w), 0, max_level);
    let zero = &solver.int_var(0, 0);

    for lv in 0..=max_level {
        let on_level = &(levels.eq(lv));
        let is_seed = &solver.bool_var_2d((h, w));

        if lv == 0 {
            for y in 0..h {
                for x in 0..w {
                    let mut cands = vec![];
                    if y == 0 {
                        cands.push(!is_line.horizontal.at((y, x)));
                    }
                    if y == h - 1 {
                        cands.push(!is_line.horizontal.at((y + 1, x)));
                    }
                    if x == 0 {
                        cands.push(!is_line.vertical.at((y, x)));
                    }
                    if x == w - 1 {
                        cands.push(!is_line.vertical.at((y, x + 1)));
                    }
                    solver.add_expr(is_seed.at((y, x)).iff(any(cands)));
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let cands = [
                        is_line.horizontal.at((y, x))
                            & levels.at_offset((y, x), (-1, 0), zero).eq(lv - 1),
                        is_line.horizontal.at((y + 1, x))
                            & levels.at_offset((y, x), (1, 0), zero).eq(lv - 1),
                        is_line.vertical.at((y, x))
                            & levels.at_offset((y, x), (0, -1), zero).eq(lv - 1),
                        is_line.vertical.at((y, x + 1))
                            & levels.at_offset((y, x), (0, 1), zero).eq(lv - 1),
                    ];
                    solver.add_expr(
                        is_seed
                            .at((y, x))
                            .iff(any(cands) & levels.at((y, x)).ge(lv)),
                    );
                }
            }
        }
        solver.add_expr(is_seed.imp(on_level));
        solver.add_expr(
            is_line
                .horizontal
                .slice((1..h, ..))
                .imp(!(on_level.slice((..(h - 1), ..)) & on_level.slice((1.., ..)))),
        );
        solver.add_expr(
            (!is_line.horizontal.slice((1..h, ..))).imp(
                on_level
                    .slice((..(h - 1), ..))
                    .iff(on_level.slice((1.., ..))),
            ),
        );
        solver.add_expr(
            is_line
                .vertical
                .slice((.., 1..w))
                .imp(!(on_level.slice((.., ..(w - 1))) & on_level.slice((.., 1..)))),
        );
        solver.add_expr(
            (!is_line.vertical.slice((.., 1..w))).imp(
                on_level
                    .slice((.., ..(w - 1)))
                    .iff(on_level.slice((.., 1..))),
            ),
        );

        let mut is_active = vec![];
        for y in 0..h {
            for x in 0..w {
                is_active.push(on_level.at((y, x)));
            }
        }
        for y in 0..h {
            for x in 0..w {
                is_active.push(is_seed.at((y, x)).expr());
            }
        }
        is_active.push(TRUE);
        graph::active_vertices_connected(&mut solver, is_active, &aux_graph);
    }
    for y in 0..h {
        for x in 0..w {
            if let Some((_, l)) = clues[y][x] {
                if l >= 0 {
                    solver.add_expr(levels.at((y, x)).eq(l));
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

type Problem = Vec<Vec<Option<(i32, i32)>>>;

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let desc = get_kudamono_url_info(url)?;
    if desc.puzzle_kind != "crosswall" {
        return None;
    }
    let mut ret = vec![vec![None; desc.width]; desc.height];
    let content = desc.content.as_bytes();
    let mut sequencer = Sequencer::new(content);
    let mut pos = 0;
    let ctx = Context::new();
    let y_ord0 = lexicographic_order(desc.height);
    let y_ord2 = lexicographic_order(desc.height - 1);

    let x_ord_base = lexicographic_order(desc.width);
    let mut x_ord = vec![(0, 0)];
    for n in x_ord_base {
        // n, n + eps, n + 1 - eps
        x_ord.push((n, 1));
        x_ord.push((n, 2));
        if n != desc.width - 1 {
            x_ord.push((n + 1, 0));
        }
    }

    while sequencer.n_read() < content.len() {
        if sequencer.peek() != Some('(' as u8) {
            return None;
        }
        let val = sequencer.deserialize(&ctx, DecInt)?;
        assert_eq!(val.len(), 1);
        let val = val[0];
        if sequencer.peek() != Some(')' as u8) {
            return None;
        }
        let ofs = sequencer.deserialize(&ctx, DecInt)?;
        assert_eq!(ofs.len(), 1);
        let ofs = ofs[0];
        pos += ofs as usize;
        let (x, xm) = x_ord[pos / desc.height];

        if xm == 0 {
            let y = desc.height - 1 - y_ord0[pos % desc.height];
            ret[y][x] = Some((
                val,
                match ret[y][x] {
                    Some((_, t)) => t,
                    _ => -1,
                },
            ));
        } else if xm == 2 {
            let y = pos % desc.height;
            let y = desc.height - 1 - if y == 0 { 0 } else { y_ord2[y - 1] + 1 };
            ret[y][x] = Some((
                match ret[y][x] {
                    Some((t, _)) => t,
                    _ => -1,
                },
                val,
            ));
        }
    }

    Some(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![None, Some((4, 0)), None, None, None],
            vec![None, None, None, Some((-1, 2)), None],
            vec![None, Some((3, -1)), Some((-1, 2)), None, Some((6, -1))],
            vec![None, Some((2, -1)), Some((1, -1)), None, None],
            vec![None, None, Some((4, -1)), None, None],
        ]
    }

    #[test]
    fn test_crosswall_problem() {
        let clues = problem_for_tests();
        let ans = solve_crosswall(&clues);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::GridEdges {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 1],
                [1, 1, 0, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 0, 1],
                [0, 0, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_crosswall_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=4&H=4&L=(2)16(3)1(4)2(0)10(4)1(1)1(2)11(2)16(6)4&G=crosswall";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
