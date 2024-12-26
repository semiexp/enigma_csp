use crate::graph;
use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, Combinator, Context, KudamonoBorder,
};
use crate::solver::{any, count_true, Solver, FALSE};

pub fn solve_double_lits(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
) -> Option<Vec<Vec<Option<bool>>>> {
    let h = borders.vertical.len();
    assert!(h > 0);
    let w = borders.vertical[0].len() + 1;

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    graph::active_vertices_connected_2d(&mut solver, is_black);

    solver.add_expr(
        !(is_black.slice((..(h - 1), ..(w - 1)))
            & is_black.slice((..(h - 1), 1..))
            & is_black.slice((1.., ..(w - 1)))
            & is_black.slice((1.., 1..))),
    );

    let sub_boards = [&solver.bool_var_2d((h, w)), &solver.bool_var_2d((h, w))];
    solver.add_expr(!(sub_boards[0] & sub_boards[1]));
    solver.add_expr((sub_boards[0] | sub_boards[1]).iff(is_black));

    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && !borders.horizontal[y][x] {
                solver.add_expr(!(sub_boards[0].at((y, x)) & sub_boards[1].at((y + 1, x))));
                solver.add_expr(!(sub_boards[1].at((y, x)) & sub_boards[0].at((y + 1, x))));
            }
            if x < w - 1 && !borders.vertical[y][x] {
                solver.add_expr(!(sub_boards[0].at((y, x)) & sub_boards[1].at((y, x + 1))));
                solver.add_expr(!(sub_boards[1].at((y, x)) & sub_boards[0].at((y, x + 1))));
            }
        }
    }
    // 0: white cell
    // 1: endpoint
    // 2: L
    // 3: I
    // 4: T
    let kinds = [
        &solver.int_var_2d((h, w), 0, 4),
        &solver.int_var_2d((h, w), 0, 4),
    ];

    let cell_tetro_type = &solver.int_var_2d((h, w), 0, 4);

    for t in 0..2 {
        solver.add_expr(sub_boards[t].iff(kinds[t].ne(0)));

        for y in 0..h {
            for x in 0..w {
                let mut neighbors = vec![];
                if y > 0 && !borders.horizontal[y - 1][x] {
                    neighbors.push(sub_boards[t].at((y - 1, x)).expr());
                } else {
                    neighbors.push(FALSE);
                }
                if x > 0 && !borders.vertical[y][x - 1] {
                    neighbors.push(sub_boards[t].at((y, x - 1)).expr());
                } else {
                    neighbors.push(FALSE);
                }
                if y < h - 1 && !borders.horizontal[y][x] {
                    neighbors.push(sub_boards[t].at((y + 1, x)).expr());
                } else {
                    neighbors.push(FALSE);
                }
                if x < w - 1 && !borders.vertical[y][x] {
                    neighbors.push(sub_boards[t].at((y, x + 1)).expr());
                } else {
                    neighbors.push(FALSE);
                }

                solver.add_expr(kinds[t].at((y, x)).eq(1).imp(count_true(&neighbors).eq(1)));
                solver.add_expr(kinds[t].at((y, x)).eq(2).imp(any([
                    &neighbors[0] & &neighbors[1] & !&neighbors[2] & !&neighbors[3],
                    &neighbors[1] & &neighbors[2] & !&neighbors[3] & !&neighbors[0],
                    &neighbors[2] & &neighbors[3] & !&neighbors[0] & !&neighbors[1],
                    &neighbors[3] & &neighbors[0] & !&neighbors[1] & !&neighbors[2],
                ])));
                solver.add_expr(kinds[t].at((y, x)).eq(3).imp(any([
                    &neighbors[0] & &neighbors[2] & !&neighbors[1] & !&neighbors[3],
                    &neighbors[1] & &neighbors[3] & !&neighbors[0] & !&neighbors[2],
                ])));
                solver.add_expr(kinds[t].at((y, x)).eq(4).imp(count_true(&neighbors).eq(3)));
            }
        }

        let rooms = graph::borders_to_rooms(borders);
        for room in &rooms {
            let mut cell_kinds = vec![vec![]; 5];
            let room_kind = solver.int_var(0, 3);
            for &(y, x) in room {
                for j in 1..=4 {
                    cell_kinds[j].push(kinds[t].at((y, x)).eq(j as i32));
                }
            }
            let cell_kind_counts = cell_kinds.iter().map(|x| count_true(x)).collect::<Vec<_>>();
            // L
            solver.add_expr(room_kind.eq(0).imp(
                cell_kind_counts[1].eq(2)
                    & cell_kind_counts[2].eq(1)
                    & cell_kind_counts[3].eq(1)
                    & cell_kind_counts[4].eq(0),
            ));
            // I
            solver.add_expr(room_kind.eq(1).imp(
                cell_kind_counts[1].eq(2)
                    & cell_kind_counts[2].eq(0)
                    & cell_kind_counts[3].eq(2)
                    & cell_kind_counts[4].eq(0),
            ));
            // T
            solver.add_expr(room_kind.eq(2).imp(
                cell_kind_counts[1].eq(3)
                    & cell_kind_counts[2].eq(0)
                    & cell_kind_counts[3].eq(0)
                    & cell_kind_counts[4].eq(1),
            ));
            // S
            solver.add_expr(room_kind.eq(3).imp(
                cell_kind_counts[1].eq(2)
                    & cell_kind_counts[2].eq(2)
                    & cell_kind_counts[3].eq(0)
                    & cell_kind_counts[4].eq(0),
            ));

            for &(y, x) in room {
                solver.add_expr(
                    sub_boards[t]
                        .at((y, x))
                        .imp(cell_tetro_type.at((y, x)).eq(&room_kind)),
                );
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && borders.horizontal[y][x] {
                solver.add_expr(
                    (is_black.at((y, x)) & is_black.at((y + 1, x))).imp(
                        cell_tetro_type
                            .at((y, x))
                            .ne(cell_tetro_type.at((y + 1, x))),
                    ),
                );
            }
            if x < w - 1 && borders.vertical[y][x] {
                solver.add_expr(
                    (is_black.at((y, x)) & is_black.at((y, x + 1))).imp(
                        cell_tetro_type
                            .at((y, x))
                            .ne(cell_tetro_type.at((y, x + 1))),
                    ),
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

type Problem = graph::InnerGridEdges<Vec<Vec<bool>>>;

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W")?)?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    let border;
    if let Some(p) = parsed.get("SIE") {
        border = KudamonoBorder.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        border = graph::InnerGridEdges {
            horizontal: vec![vec![false; width]; height - 1],
            vertical: vec![vec![false; width - 1]; height],
        };
    }

    Some(border)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        graph::InnerGridEdges {
            horizontal: crate::puzzle::util::tests::to_bool_2d([
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_bool_2d([
                [0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]),
        }
    }

    #[test]
    fn test_double_lits() {
        let problem = problem_for_tests();
        let ans = solve_double_lits(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [1, 1, 0, 2, 1, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 2],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 2],
            [1, 1, 1, 0, 2, 0],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_double_lits_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=6x6&SIE=19U3LLUUUURRRDRDLLDDD&G=lits&V=double";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
