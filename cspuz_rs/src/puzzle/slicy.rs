use crate::graph;
use crate::hex::{borders_to_rooms, BoolHexGridIrrefutableFacts, HexGrid, HexInnerGridEdges};
use crate::serializer::get_kudamono_url_info_detailed;
use crate::solver::{all, any, Solver};

pub fn solve_slicy(borders: &HexInnerGridEdges<bool>) -> Option<BoolHexGridIrrefutableFacts> {
    let mut solver = Solver::new();
    let is_black = &HexGrid::new(&mut solver, borders.dims);
    solver.add_answer_key_bool(is_black.flatten());

    let (is_black_flat, g) = is_black.representation();
    graph::active_vertices_connected(&mut solver, &is_black_flat, &g);

    let rooms = borders_to_rooms(borders);
    let room_shape_type = &solver.int_var_1d(rooms.len(), 0, 4);

    let piece_variants = get_piece_variants();

    let (height, width) = is_black.repr_dims();
    let mut cell_to_room_id = vec![vec![!0; width]; height];
    for i in 0..rooms.len() {
        for &(y, x) in &rooms[i] {
            cell_to_room_id[y][x] = i;
        }
    }

    for &(y, x) in is_black.cells() {
        if !is_black.is_valid_coord_offset((y, x), (1, 1)) {
            continue;
        }

        if is_black.is_valid_coord_offset((y, x), (1, 0)) {
            solver
                .add_expr(!(&is_black[(y, x)] & &is_black[(y + 1, x)] & &is_black[(y + 1, x + 1)]));
        }
        if is_black.is_valid_coord_offset((y, x), (0, 1)) {
            solver
                .add_expr(!(&is_black[(y, x)] & &is_black[(y, x + 1)] & &is_black[(y + 1, x + 1)]));
        }
    }

    for &(y, x) in is_black.cells() {
        if is_black.is_valid_coord_offset((y, x), (0, 1)) && borders.to_right[(y, x)] {
            solver.add_expr(
                (&is_black[(y, x)] & &is_black[(y, x + 1)]).imp(
                    room_shape_type
                        .at(cell_to_room_id[y][x])
                        .ne(room_shape_type.at(cell_to_room_id[y][x + 1])),
                ),
            );
        }
        if is_black.is_valid_coord_offset((y, x), (1, 0)) && borders.to_bottom_left[(y, x)] {
            solver.add_expr(
                (&is_black[(y, x)] & &is_black[(y + 1, x)]).imp(
                    room_shape_type
                        .at(cell_to_room_id[y][x])
                        .ne(room_shape_type.at(cell_to_room_id[y + 1][x])),
                ),
            );
        }
        if is_black.is_valid_coord_offset((y, x), (1, 1)) && borders.to_bottom_right[(y, x)] {
            solver.add_expr(
                (&is_black[(y, x)] & &is_black[(y + 1, x + 1)]).imp(
                    room_shape_type
                        .at(cell_to_room_id[y][x])
                        .ne(room_shape_type.at(cell_to_room_id[y + 1][x + 1])),
                ),
            );
        }
    }

    for i in 0..rooms.len() {
        let room = &rooms[i];

        for s in 0..5 {
            let mut candidates = vec![];

            for variant in &piece_variants[s] {
                for &(ty, tx) in room {
                    let mut isok = true;

                    for &(dy, dx) in variant {
                        if !is_black.is_valid_coord_offset((ty, tx), (dy, dx)) {
                            isok = false;
                            break;
                        }
                        if cell_to_room_id[(ty as i32 + dy) as usize][(tx as i32 + dx) as usize]
                            != i
                        {
                            isok = false;
                            break;
                        }
                    }

                    if !isok {
                        continue;
                    }

                    let mut clause = vec![];
                    let piece = variant
                        .iter()
                        .map(|&(dy, dx)| ((ty as i32 + dy) as usize, (tx as i32 + dx) as usize))
                        .collect::<Vec<_>>();

                    for &(y, x) in room {
                        if !piece.contains(&(y, x)) {
                            clause.push(!&is_black[(y, x)]);
                        } else {
                            clause.push(is_black[(y, x)].expr());
                        }
                    }

                    candidates.push(all(clause));
                }
            }

            solver.add_expr(room_shape_type.at(i).eq(s as i32).imp(any(candidates)));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

fn get_piece_variants() -> Vec<Vec<Vec<(i32, i32)>>> {
    vec![
        vec![
            vec![(0, 0), (0, 1), (1, -1), (1, 0)],
            vec![(0, 0), (0, 1), (1, 2), (1, 3)],
            vec![(0, 0), (1, -1), (1, 0), (2, -1)],
            vec![(0, 0), (1, 0), (2, 1), (3, 1)],
            vec![(0, 0), (1, 1), (1, 2), (2, 3)],
            vec![(0, 0), (1, 1), (2, 1), (3, 2)],
        ],
        vec![
            vec![(0, 0), (0, 1), (0, 2), (1, 0)],
            vec![(0, 0), (0, 1), (0, 2), (1, 3)],
            vec![(0, 0), (0, 1), (1, 0), (2, 0)],
            vec![(0, 0), (0, 1), (1, 2), (2, 3)],
            vec![(0, 0), (1, -2), (1, -1), (1, 0)],
            vec![(0, 0), (1, 0), (2, -1), (2, 0)],
            vec![(0, 0), (1, 0), (2, 0), (3, 1)],
            vec![(0, 0), (1, 0), (2, 1), (3, 2)],
            vec![(0, 0), (1, 1), (1, 2), (1, 3)],
            vec![(0, 0), (1, 1), (2, 1), (3, 1)],
            vec![(0, 0), (1, 1), (2, 2), (2, 3)],
            vec![(0, 0), (1, 1), (2, 2), (3, 2)],
        ],
        vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3)],
            vec![(0, 0), (1, 0), (2, 0), (3, 0)],
            vec![(0, 0), (1, 1), (2, 2), (3, 3)],
        ],
        vec![
            vec![(0, 0), (0, 1), (1, 0), (1, 2)],
            vec![(0, 0), (0, 1), (1, 0), (2, 1)],
            vec![(0, 0), (0, 1), (1, 2), (2, 2)],
            vec![(0, 0), (0, 2), (1, 1), (1, 2)],
            vec![(0, 0), (1, 0), (2, 1), (2, 2)],
            vec![(0, 0), (1, 1), (2, 0), (2, 1)],
        ],
        vec![
            vec![(0, 0), (1, -1), (1, 0), (2, 1)],
            vec![(0, 0), (1, 1), (1, 2), (2, 1)],
        ],
    ]
}

type Problem = HexInnerGridEdges<bool>;

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let dims = {
        let mut it = parsed["W"].split('x');
        let a: usize = it.next()?.parse().ok()?;
        let b: usize = it.next()?.parse().ok()?;
        let c: usize = it.next()?.parse().ok()?;
        if it.next().is_some() {
            return None;
        }
        (c, a, b, b)
    };

    let mut ret = HexInnerGridEdges {
        dims,
        to_right: HexGrid::filled(dims, false),
        to_bottom_left: HexGrid::filled(dims, false),
        to_bottom_right: HexGrid::filled(dims, false),
    };

    let to_brick_coord = |y: usize, x: usize| -> (usize, usize) {
        if y >= dims.0 {
            (y, x * 2 - (y - (dims.0 - 1)))
        } else {
            (y, x * 2 + (dims.0 - 1 - y))
        }
    };

    let mut vertices = vec![];
    let mut edges = vec![];

    for &(y, x) in ret.to_right.cells() {
        let (by, bx) = to_brick_coord(y, x);

        for dy in 0..2 {
            for dx in 0..3 {
                vertices.push((by + dy, bx + dx));
            }
        }

        if ret.to_right.is_valid_coord_offset((y, x), (0, 1)) {
            edges.push(((by, bx + 2), (by + 1, bx + 2), (y, x), 0));
        }
        if ret.to_right.is_valid_coord_offset((y, x), (1, 0)) {
            edges.push(((by + 1, bx), (by + 1, bx + 1), (y, x), 1));
        }
        if ret.to_right.is_valid_coord_offset((y, x), (1, 1)) {
            edges.push(((by + 1, bx + 1), (by + 1, bx + 2), (y, x), 2));
        }
    }
    vertices.sort_by(|&(y1, x1), &(y2, x2)| {
        if x1 < x2 {
            std::cmp::Ordering::Less
        } else if x1 > x2 {
            std::cmp::Ordering::Greater
        } else if y1 > y2 {
            std::cmp::Ordering::Less
        } else if y1 < y2 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    vertices.dedup();
    edges.sort();

    {
        let input = parsed.get("SIE")?.as_bytes();
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
                let (mut y, mut x) = vertices[pos];

                while idx < input.len() {
                    let (dy, dx) = match input[idx] {
                        b'L' | b'W' => (0, -1),
                        b'R' | b'E' => (0, 1),
                        b'U' => (-1, 0),
                        b'D' => (1, 0),
                        _ => break,
                    };

                    if y as i32 + dy < 0 || x as i32 + dx < 0 {
                        return None;
                    }

                    let (y2, x2) = ((y as i32 + dy) as usize, (x as i32 + dx) as usize);

                    let (pa, pb) = if (y, x) < (y2, x2) {
                        ((y, x), (y2, x2))
                    } else {
                        ((y2, x2), (y, x))
                    };

                    let edge_id = edges.binary_search(&(pa, pb, (0, 0), -1)).err()?;
                    if edge_id >= edges.len() || edges[edge_id].0 != pa || edges[edge_id].1 != pb {
                        return None;
                    }

                    match edges[edge_id].3 {
                        0 => {
                            ret.to_right[edges[edge_id].2] = true;
                        }
                        1 => {
                            ret.to_bottom_left[edges[edge_id].2] = true;
                        }
                        2 => {
                            ret.to_bottom_right[edges[edge_id].2] = true;
                        }
                        _ => unreachable!(),
                    }

                    y = y2;
                    x = x2;
                    idx += 1;
                }
            }
        }
    }

    Some(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_piece(piece: &[(i32, i32)]) -> Vec<(i32, i32)> {
        let mut piece = piece.to_vec();
        piece.sort();

        let y0 = piece[0].0;
        let x0 = piece[0].1;

        for p in &mut piece {
            p.0 -= y0;
            p.1 -= x0;
        }
        piece
    }

    fn flip_piece(piece: &[(i32, i32)]) -> Vec<(i32, i32)> {
        let piece = piece.iter().map(|&(y, x)| (x, y)).collect::<Vec<_>>();
        normalize_piece(&piece)
    }

    fn rotate_piece(piece: &[(i32, i32)]) -> Vec<(i32, i32)> {
        // (y, x) -> (x, x - y) -> (x - y, -y) -> (-y, -x) -> (-x, -x + y) -> (-x + y, y) -> (y, x)
        let piece = piece.iter().map(|&(y, x)| (x, x - y)).collect::<Vec<_>>();
        normalize_piece(&piece)
    }

    fn compute_piece_variants(piece: &[(i32, i32)]) -> Vec<Vec<(i32, i32)>> {
        let mut variants = vec![];
        let mut piece = normalize_piece(&piece);
        for _ in 0..6 {
            variants.push(piece.clone());
            variants.push(flip_piece(&piece));
            piece = rotate_piece(&piece);
        }
        variants.sort();
        variants.dedup();
        variants
    }

    #[test]
    fn test_piece_variants() {
        let actual = get_piece_variants();
        let expected = vec![
            compute_piece_variants(&[(0, 0), (0, 1), (1, 2), (1, 3)]), // S
            compute_piece_variants(&[(0, 0), (0, 1), (0, 2), (1, 3)]), // L
            compute_piece_variants(&[(0, 0), (0, 1), (0, 2), (0, 3)]), // I
            compute_piece_variants(&[(0, 0), (0, 1), (1, 0), (1, 2)]), // C
            compute_piece_variants(&[(0, 0), (1, 1), (1, 2), (2, 1)]), // Y
        ];

        assert_eq!(actual.len(), expected.len());

        for (mut actual, expected) in actual.into_iter().zip(expected) {
            actual.sort();
            assert_eq!(actual, expected);
        }
    }

    fn problem_for_tests() -> HexInnerGridEdges<bool> {
        HexInnerGridEdges {
            dims: (5, 3, 4, 4),
            to_right: HexGrid::from_grid(
                (5, 3, 4, 4),
                crate::puzzle::util::tests::to_bool_2d([
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]),
            ),
            to_bottom_left: HexGrid::from_grid(
                (5, 3, 4, 4),
                crate::puzzle::util::tests::to_bool_2d([
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]),
            ),
            to_bottom_right: HexGrid::from_grid(
                (5, 3, 4, 4),
                crate::puzzle::util::tests::to_bool_2d([
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]),
            ),
        }
    }

    #[test]
    fn test_slicy_problem() {
        let borders = problem_for_tests();
        let ans = solve_slicy(&borders);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = HexGrid::from_grid(
            (5, 3, 4, 4),
            crate::puzzle::util::tests::to_option_bool_2d([
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 2, 0],
                [1, 0, 0, 1, 1, 2],
                [1, 0, 2, 1, 0, 0],
                [0, 1, 0, 0, 1, 2],
                [0, 0, 1, 1, 2, 2],
                [0, 0, 0, 0, 2, 1],
            ]),
        );
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_slicy_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=3x4x5&SIE=4REUEUEUEU25UEULWLULU6RDREUERE5EUERER&G=slicy";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
