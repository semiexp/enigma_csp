use crate::graph;
use crate::solver::{any, count_true, Solver};

pub fn solve_lits(
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

    // 0: white cell
    // 1: endpoint
    // 2: L
    // 3: I
    // 4: T
    let kind = &solver.int_var_2d((h, w), 0, 4);
    solver.add_expr(is_black.iff(kind.ne(0)));

    for y in 0..h {
        for x in 0..w {
            let mut neighbors = vec![];
            if y > 0 && !borders.horizontal[y - 1][x] {
                neighbors.push(is_black.at((y - 1, x)).any());
            } else {
                neighbors.push(any([false]));
            }
            if x > 0 && !borders.vertical[y][x - 1] {
                neighbors.push(is_black.at((y, x - 1)).any());
            } else {
                neighbors.push(any([false]));
            }
            if y < h - 1 && !borders.horizontal[y][x] {
                neighbors.push(is_black.at((y + 1, x)).any());
            } else {
                neighbors.push(any([false]));
            }
            if x < w - 1 && !borders.vertical[y][x] {
                neighbors.push(is_black.at((y, x + 1)).any());
            } else {
                neighbors.push(any([false]));
            }

            solver.add_expr(kind.at((y, x)).eq(1).imp(count_true(&neighbors).eq(1)));
            solver.add_expr(kind.at((y, x)).eq(2).imp(any([
                &neighbors[0] & &neighbors[1] & !&neighbors[2] & !&neighbors[3],
                &neighbors[1] & &neighbors[2] & !&neighbors[3] & !&neighbors[0],
                &neighbors[2] & &neighbors[3] & !&neighbors[0] & !&neighbors[1],
                &neighbors[3] & &neighbors[0] & !&neighbors[1] & !&neighbors[2],
            ])));
            solver.add_expr(kind.at((y, x)).eq(3).imp(any([
                &neighbors[0] & &neighbors[2] & !&neighbors[1] & !&neighbors[3],
                &neighbors[1] & &neighbors[3] & !&neighbors[0] & !&neighbors[2],
            ])));
            solver.add_expr(kind.at((y, x)).eq(4).imp(count_true(&neighbors).eq(3)));
        }
    }

    let rooms = graph::borders_to_rooms(borders);
    let mut room_id = vec![vec![0; w]; h];
    let room_kind = &solver.int_var_1d(rooms.len(), 0, 3);
    for (i, room) in rooms.iter().enumerate() {
        let mut cell_kinds = vec![vec![]; 5];
        for &(y, x) in room {
            room_id[y][x] = i;
            for j in 1..=4 {
                cell_kinds[j].push(kind.at((y, x)).eq(j as i32));
            }
        }
        let cell_kind_counts = cell_kinds.iter().map(|x| count_true(x)).collect::<Vec<_>>();
        // L
        solver.add_expr(room_kind.at(i).eq(0).imp(
            cell_kind_counts[1].eq(2)
                & cell_kind_counts[2].eq(1)
                & cell_kind_counts[3].eq(1)
                & cell_kind_counts[4].eq(0),
        ));
        // I
        solver.add_expr(room_kind.at(i).eq(1).imp(
            cell_kind_counts[1].eq(2)
                & cell_kind_counts[2].eq(0)
                & cell_kind_counts[3].eq(2)
                & cell_kind_counts[4].eq(0),
        ));
        // T
        solver.add_expr(room_kind.at(i).eq(2).imp(
            cell_kind_counts[1].eq(3)
                & cell_kind_counts[2].eq(0)
                & cell_kind_counts[3].eq(0)
                & cell_kind_counts[4].eq(1),
        ));
        // S
        solver.add_expr(room_kind.at(i).eq(3).imp(
            cell_kind_counts[1].eq(2)
                & cell_kind_counts[2].eq(2)
                & cell_kind_counts[3].eq(0)
                & cell_kind_counts[4].eq(0),
        ));
    }
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && room_id[y][x] != room_id[y + 1][x] {
                solver.add_expr(
                    (is_black.at((y, x)) & is_black.at((y + 1, x))).imp(
                        room_kind
                            .at(room_id[y][x])
                            .ne(room_kind.at(room_id[y + 1][x])),
                    ),
                );
            }
            if x < w - 1 && room_id[y][x] != room_id[y][x + 1] {
                solver.add_expr(
                    (is_black.at((y, x)) & is_black.at((y, x + 1))).imp(
                        room_kind
                            .at(room_id[y][x])
                            .ne(room_kind.at(room_id[y][x + 1])),
                    ),
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> graph::InnerGridEdges<Vec<Vec<bool>>> {
        // https://github.com/semiexp/cspuz/blob/d8d6df349c6c96500a996a46e3810977b513b3de/cspuz/puzzle/lits.py#L290-L301
        let height = 10;
        let width = 10;
        let base = [
            "0000000222",
            "0010002222",
            "1113332222",
            "5563444288",
            "5663422228",
            "5663223338",
            "5633333338",
            "6673339aaa",
            "6773999aab",
            "77bbbbbbbb",
        ];
        let base = base.map(|x| x.chars().collect::<Vec<_>>());
        let mut horizontal = vec![vec![false; width]; height - 1];
        let mut vertical = vec![vec![false; width - 1]; height];
        for y in 0..height {
            for x in 0..width {
                if y < height - 1 {
                    horizontal[y][x] = base[y][x] != base[y + 1][x];
                }
                if x < width - 1 {
                    vertical[y][x] = base[y][x] != base[y][x + 1];
                }
            }
        }
        graph::InnerGridEdges {
            horizontal,
            vertical,
        }
    }

    #[test]
    fn test_lits_problem() {
        let problem = problem_for_tests();
        let ans = solve_lits(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_base = [
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        ];
        let expected =
            expected_base.map(|row| row.iter().map(|&n| Some(n == 1)).collect::<Vec<_>>());
        assert_eq!(ans, expected);
    }
}
