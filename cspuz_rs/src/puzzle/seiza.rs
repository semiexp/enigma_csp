use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, Combinator, Context, DecInt, Dict,
    KudamonoBorder, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::{count_true, Solver};

pub fn solve_seiza(
    absent_cell: &[Vec<bool>],
    num: &[Vec<Option<i32>>],
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
) -> Option<(graph::BoolGridEdgesIrrefutableFacts, Vec<Vec<Option<bool>>>)> {
    let (h, w) = util::infer_shape(absent_cell);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_star = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_star);

    solver.add_expr(!(is_star.slice((..(h - 1), ..)) & is_star.slice((1.., ..))));
    solver.add_expr(!(is_star.slice((.., ..(w - 1))) & is_star.slice((.., 1..))));
    solver.add_expr(!(is_star.slice((..(h - 1), ..(w - 1))) & is_star.slice((1.., 1..))));
    solver.add_expr(!(is_star.slice((..(h - 1), 1..)) & is_star.slice((1.., ..(w - 1)))));

    for y in 0..h {
        for x in 0..w {
            if absent_cell[y][x] {
                solver.add_expr(!is_star.at((y, x)));
                solver.add_expr(!(is_line.vertex_neighbors((y, x)).any()));
            } else {
                if y == 0 {
                    solver.add_expr((!is_star.at((y, x))).imp(!is_line.vertical.at((y, x))));
                } else if y == h - 1 {
                    solver.add_expr((!is_star.at((y, x))).imp(!is_line.vertical.at((y - 1, x))));
                } else {
                    solver.add_expr(
                        (!is_star.at((y, x))).imp(
                            is_line
                                .vertical
                                .at((y, x))
                                .iff(is_line.vertical.at((y - 1, x))),
                        ),
                    );
                }

                if x == 0 {
                    solver.add_expr((!is_star.at((y, x))).imp(!is_line.horizontal.at((y, x))));
                } else if x == w - 1 {
                    solver.add_expr((!is_star.at((y, x))).imp(!is_line.horizontal.at((y, x - 1))));
                } else {
                    solver.add_expr(
                        (!is_star.at((y, x))).imp(
                            is_line
                                .horizontal
                                .at((y, x))
                                .iff(is_line.horizontal.at((y, x - 1))),
                        ),
                    );
                }

                solver
                    .add_expr((!is_star.at((y, x))).imp(!(is_line.vertex_neighbors((y, x)).all())));
                solver.add_expr(
                    is_star
                        .at((y, x))
                        .imp(is_line.vertex_neighbors((y, x)).any()),
                );
            }
        }
    }

    let mut borders = borders.clone();
    for y in 0..h {
        for x in 0..w {
            if absent_cell[y][x] {
                if y > 0 {
                    borders.horizontal[y - 1][x] = true;
                }
                if y + 1 < h {
                    borders.horizontal[y][x] = true;
                }
                if x > 0 {
                    borders.vertical[y][x - 1] = true;
                }
                if x + 1 < w {
                    borders.vertical[y][x] = true;
                }
            }
        }
    }

    let rooms = graph::borders_to_rooms(&borders);
    for room in rooms {
        let room = room
            .into_iter()
            .filter(|&(y, x)| !absent_cell[y][x])
            .collect::<Vec<_>>();
        if room.is_empty() {
            continue;
        }

        // exactly one star
        let mut constr = vec![];
        for &(y, x) in &room {
            constr.push(is_star.at((y, x)));
        }
        solver.add_expr(count_true(&constr).eq(1));

        let mut n = None;
        for &(y, x) in &room {
            if let Some(c) = num[y][x] {
                if let Some(cc) = n {
                    if cc != c {
                        return None;
                    }
                } else {
                    n = Some(c);
                }
            }
        }

        if let Some(n) = n {
            for &(y, x) in &room {
                solver.add_expr(
                    is_star
                        .at((y, x))
                        .imp(is_line.vertex_neighbors((y, x)).count_true().eq(n)),
                );
            }
        }
    }

    let (is_line_flat, conn) = is_line.representation();
    graph::active_vertices_connected(&mut solver, is_line_flat, &conn.line_graph());

    solver
        .irrefutable_facts()
        .map(|f| (f.get(is_line), f.get(is_star)))
}

pub type Problem = (
    Vec<Vec<bool>>,
    Vec<Vec<Option<i32>>>,
    graph::InnerGridEdges<Vec<Vec<bool>>>,
);

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W")?)?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    let absent_cell;
    if let Some(p) = parsed.get("L") {
        let absent_cell_combinator = KudamonoGrid::new(Dict::new(true, "x"), false);
        absent_cell = absent_cell_combinator
            .deserialize(&ctx, p.as_bytes())?
            .1
            .pop()?;
    } else {
        absent_cell = vec![vec![false; width]; height];
    }

    let num;
    if let Some(p) = parsed.get("L-N") {
        let num_combinator = KudamonoGrid::new(
            Optionalize::new(PrefixAndSuffix::new("(", DecInt, ")")),
            None,
        );
        num = num_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        num = vec![vec![None; width]; height];
    }

    let border;
    if let Some(p) = parsed.get("SIE") {
        border = KudamonoBorder.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        border = graph::InnerGridEdges {
            horizontal: vec![vec![false; width]; height - 1],
            vertical: vec![vec![false; width - 1]; height],
        };
    }

    Some((absent_cell, num, border))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        (
            crate::puzzle::util::tests::to_bool_2d([
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),
            vec![
                vec![Some(2), None, None, None, None, None, None],
                vec![None, None, None, Some(2), None, None, None],
                vec![None, None, None, None, None, None, None],
                vec![None, Some(3), None, None, None, None, None],
                vec![None, None, None, None, None, None, None],
            ],
            graph::InnerGridEdges {
                horizontal: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 0],
                ]),
                vertical: crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 1, 0],
                ]),
            },
        )
    }

    #[test]
    fn test_seiza_problem() {
        let (absent_cell, num, borders) = problem_for_tests();
        let ans = solve_seiza(&absent_cell, &num, &borders);
        assert!(ans.is_some());
        let ans: (
            graph::GridEdges<Vec<Vec<Option<bool>>>>,
            Vec<Vec<Option<bool>>>,
        ) = ans.unwrap();

        let expected_lines = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ]),
        };
        let expected_stars = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1],
        ]);

        assert_eq!(ans, (expected_lines, expected_stars));
    }

    #[test]
    fn test_seiza_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player.html?W=7x5&L=x9&L-N=(2)4(3)2(2)12&SIE=3RU5RRDD11RRD1URRRUU3DDLLDDD8URR8R&G=seiza";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
