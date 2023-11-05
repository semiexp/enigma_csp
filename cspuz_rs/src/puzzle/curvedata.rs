use super::util;
use crate::graph::{self, BoolGridEdges};
use crate::serializer::{
    strip_prefix, Choice, Combinator, Context, ContextBasedGrid, Dict, HexInt, Map, Rooms, Spaces,
};
use crate::solver::{IntVarArray2D, Solver};

type AdjacencyEntry = Option<(usize, usize)>;

struct Adjacency {
    up: AdjacencyEntry,
    down: AdjacencyEntry,
    left: AdjacencyEntry,
    right: AdjacencyEntry,
}

impl Adjacency {
    fn new() -> Adjacency {
        Adjacency {
            up: None,
            down: None,
            left: None,
            right: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceId {
    None,
    Block,
    Piece(usize),
}

pub fn solve_curvedata(
    piece_id: &[Vec<PieceId>],
    borders: &Option<graph::InnerGridEdges<Vec<Vec<bool>>>>,
    pieces: &[graph::GridEdges<Vec<Vec<bool>>>],
) -> Option<graph::BoolGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(piece_id);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    add_constraints(&mut solver, is_line, piece_id, borders, pieces);

    solver.irrefutable_facts().map(|f| f.get(is_line))
}

pub fn enumerate_answers_curvedata(
    piece_id: &[Vec<PieceId>],
    borders: &Option<graph::InnerGridEdges<Vec<Vec<bool>>>>,
    pieces: &[graph::GridEdges<Vec<Vec<bool>>>],
    num_max_answers: usize,
) -> Vec<graph::BoolGridEdgesModel> {
    let (h, w) = util::infer_shape(piece_id);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    add_constraints(&mut solver, is_line, piece_id, borders, pieces);

    solver
        .answer_iter()
        .take(num_max_answers)
        .map(|f| f.get_unwrap(is_line))
        .collect()
}

pub fn add_constraints(
    solver: &mut Solver,
    is_line: &graph::BoolGridEdges,
    piece_id: &[Vec<PieceId>],
    borders: &Option<graph::InnerGridEdges<Vec<Vec<bool>>>>,
    pieces: &[graph::GridEdges<Vec<Vec<bool>>>],
) {
    let (h, w) = util::infer_shape(piece_id);
    if let Some(borders) = borders {
        for y in 0..h {
            for x in 0..w {
                if y < h - 1 && borders.horizontal[y][x] {
                    solver.add_expr(!is_line.vertical.at((y, x)));
                }
                if x < w - 1 && borders.vertical[y][x] {
                    solver.add_expr(!is_line.horizontal.at((y, x)));
                }
            }
        }
    }

    let mut cell_kind: Vec<Adjacency> = vec![];
    let mut is_endpoint = vec![];
    let mut idx_constraints: Vec<(usize, usize, usize, usize)> = vec![]; // (y, x, start, end)

    cell_kind.push(Adjacency::new());
    is_endpoint.push(false);

    for y in 0..h {
        for x in 0..w {
            match piece_id[y][x] {
                PieceId::None => (),
                PieceId::Block => {
                    idx_constraints.push((y, x, 0, 1));
                }
                PieceId::Piece(id) => {
                    let piece = &pieces[id];
                    let ph = piece.vertical.len();
                    let pw = piece.horizontal[0].len();

                    let idx_start = cell_kind.len();

                    let mut pt_id: Vec<Vec<Option<usize>>> = vec![vec![None; pw + 1]; ph + 1];
                    for s in 0..=ph {
                        for t in 0..=pw {
                            let up = s != 0 && piece.vertical[s - 1][t];
                            let down = s != ph && piece.vertical[s][t];
                            let left = t != 0 && piece.horizontal[s][t - 1];
                            let right = t != pw && piece.horizontal[s][t];

                            if (!up && !down && !left && !right)
                                || (!up && !down && left && right)
                                || (up && down && !left && !right)
                            {
                                continue;
                            }

                            pt_id[s][t] = Some(cell_kind.len());
                            cell_kind.push(Adjacency::new());
                            is_endpoint.push(true);
                        }
                    }

                    for s in 0..=ph {
                        for t in 0..=pw {
                            if let Some(n) = pt_id[s][t] {
                                if s != ph && piece.vertical[s][t] {
                                    let mut s2 = s + 1;
                                    while pt_id[s2][t].is_none() {
                                        assert!(piece.vertical[s2 - 1][t]);
                                        s2 += 1;
                                    }
                                    let m = pt_id[s2][t].unwrap();
                                    let e = cell_kind.len();
                                    cell_kind.push(Adjacency::new());
                                    is_endpoint.push(false);

                                    cell_kind[n].down = Some((m, e));
                                    cell_kind[m].up = Some((n, e));
                                    cell_kind[e].down = Some((m, e));
                                    cell_kind[e].up = Some((n, e));
                                }
                                if t != pw && piece.horizontal[s][t] {
                                    let mut t2 = t + 1;
                                    while pt_id[s][t2].is_none() {
                                        assert!(piece.horizontal[s][t2 - 1]);
                                        t2 += 1;
                                    }
                                    let m = pt_id[s][t2].unwrap();
                                    let e = cell_kind.len();
                                    cell_kind.push(Adjacency::new());
                                    is_endpoint.push(false);

                                    cell_kind[n].right = Some((m, e));
                                    cell_kind[m].left = Some((n, e));
                                    cell_kind[e].right = Some((m, e));
                                    cell_kind[e].left = Some((n, e));
                                }
                            }
                        }
                    }

                    let idx_end = cell_kind.len();
                    idx_constraints.push((y, x, idx_start, idx_end));
                }
            }
        }
    }

    let cell_kind_val = &solver.int_var_2d((h, w), 0, cell_kind.len() as i32 - 1);
    for (y, x, lo, hi) in idx_constraints {
        solver.add_expr(cell_kind_val.at((y, x)).ge(lo as i32));
        solver.add_expr(cell_kind_val.at((y, x)).lt(hi as i32));
    }
    for i in 0..cell_kind.len() {
        if is_endpoint[i] {
            solver.add_expr(cell_kind_val.eq(i as i32).count_true().eq(1));
        }
    }

    fn add_constraints_sub(
        solver: &mut Solver,
        cell_kind_val: &IntVarArray2D,
        is_line: &BoolGridEdges,
        y: usize,
        x: usize,
        dy: i32,
        dx: i32,
        i: i32,
        adj: Option<(usize, usize)>,
    ) {
        let (h, w) = cell_kind_val.shape();
        if (dy == -1 && y == 0)
            || (dy == 1 && y + 1 == h)
            || (dx == -1 && x == 0)
            || (dx == 1 && x + 1 == w)
        {
            if adj.is_some() {
                solver.add_expr(cell_kind_val.at((y, x)).ne(i));
            }
            return;
        }
        let v1 = &cell_kind_val.at((y, x));
        let v2 = &cell_kind_val.at(((y as i32 + dy) as usize, (x as i32 + dx) as usize));
        let e = is_line.at(((y as i32 * 2 + dy) as usize, (x as i32 * 2 + dx) as usize));

        match adj {
            None => solver.add_expr(v1.eq(i).imp(!e)),
            Some((p, q)) => solver.add_expr(v1.eq(i).imp(e & (v2.eq(p as i32) | v2.eq(q as i32)))),
        }
    }

    for y in 0..h {
        for x in 0..w {
            if piece_id[y][x] != PieceId::Block {
                solver.add_expr(cell_kind_val.at((y, x)).ne(0));
            }
            for i in 0..cell_kind.len() {
                let desc = &cell_kind[i];
                let i = i as i32;
                add_constraints_sub(solver, cell_kind_val, is_line, y, x, -1, 0, i, desc.up);
                add_constraints_sub(solver, cell_kind_val, is_line, y, x, 1, 0, i, desc.down);
                add_constraints_sub(solver, cell_kind_val, is_line, y, x, 0, -1, i, desc.left);
                add_constraints_sub(solver, cell_kind_val, is_line, y, x, 0, 1, i, desc.right);
            }
        }
    }
}

type Problem = (
    Vec<Vec<PieceId>>,
    Option<graph::InnerGridEdges<Vec<Vec<bool>>>>,
    Vec<graph::GridEdges<Vec<Vec<bool>>>>,
);

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let content = strip_prefix(url)?;
    let toks = content.split("/").collect::<Vec<&str>>();
    if toks[0] != "curvedata" {
        return None;
    }
    let w = toks[1].parse::<usize>().ok()?;
    let h = toks[2].parse::<usize>().ok()?;

    let piece_id_combinator = ContextBasedGrid::new(Choice::new(vec![
        Box::new(Map::new(
            HexInt,
            |x| match x {
                PieceId::Piece(x) => Some(x as i32),
                _ => panic!(),
            },
            |x| Some(PieceId::Piece(x as usize)),
        )),
        Box::new(Spaces::new(PieceId::None, 'g')),
        Box::new(Dict::new(PieceId::Block, "=")),
    ]));
    let (_, mut piece_id) =
        piece_id_combinator.deserialize(&Context::sized(h, w), toks[3].as_bytes())?;
    assert_eq!(piece_id.len(), 1);
    let piece_id = piece_id.swap_remove(0);

    // TODO: consider problems with borders
    let borders;
    let offset;
    if toks[4].as_bytes()[0] == 'b' as u8 {
        let mut tmp = Rooms
            .deserialize(&Context::sized(h, w), &toks[4].as_bytes()[1..])?
            .1;
        assert_eq!(tmp.len(), 1);
        borders = Some(tmp.swap_remove(0));
        offset = 5;
    } else {
        borders = None;
        offset = 4;
    }

    let mut pieces = vec![];
    let n_pieces = (toks.len() - offset) / 3;
    for i in 0..n_pieces {
        let pw = toks[i * 3 + offset].parse::<usize>().ok()?;
        let ph = toks[i * 3 + offset + 1].parse::<usize>().ok()?;
        let desc = toks[i * 3 + offset + 2].as_bytes();

        let mut piece = graph::GridEdges {
            horizontal: vec![vec![false; pw - 1]; ph],
            vertical: vec![vec![false; pw]; ph - 1],
        };

        for j in 0..(pw * ph / 2) {
            let v;
            if '0' as u8 <= desc[j] && desc[j] <= '9' as u8 {
                v = (desc[j] - '0' as u8) as i32;
            } else {
                v = (desc[j] - 'a' as u8) as i32 + 10;
            }
            {
                let y = j * 2 / pw;
                let x = j * 2 % pw;
                if (v & 1) != 0 {
                    if x == pw - 1 {
                        return None;
                    }
                    piece.horizontal[y][x] = true;
                }
                if (v & 2) != 0 {
                    if y == ph - 1 {
                        return None;
                    }
                    piece.vertical[y][x] = true;
                }
            }
            {
                let y = (j * 2 + 1) / pw;
                let x = (j * 2 + 1) % pw;
                if y == ph && (v & 12) != 0 {
                    return None;
                }
                if (v & 4) != 0 {
                    if x == pw - 1 {
                        return None;
                    }
                    piece.horizontal[y][x] = true;
                }
                if (v & 8) != 0 {
                    if y == ph - 1 {
                        return None;
                    }
                    piece.vertical[y][x] = true;
                }
            }
        }
        pieces.push(piece);
    }

    Some((piece_id, borders, pieces))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        let piece_id = vec![
            vec![PieceId::Block, PieceId::None, PieceId::None, PieceId::None],
            vec![PieceId::None, PieceId::None, PieceId::None, PieceId::None],
            vec![PieceId::None, PieceId::Piece(0), PieceId::Piece(1), PieceId::None],
            vec![PieceId::None, PieceId::None, PieceId::None, PieceId::None],
            vec![PieceId::None, PieceId::None, PieceId::None, PieceId::None],
        ];
        let borders = Some(graph::InnerGridEdges {
            vertical: vec![
                vec![false, false, false],
                vec![false, false, false],
                vec![false, false, false],
                vec![true, false, false],
                vec![false, false, false],
            ],
            horizontal: vec![
                vec![false, false, false, false],
                vec![false, false, false, false],
                vec![false, false, false, false],
                vec![false, false, false, false],
            ],
        });
        let pieces = vec![
            graph::GridEdges {
                horizontal: vec![
                    vec![false, true],
                    vec![true, false],
                    vec![false, true],
                ],
                vertical: vec![
                    vec![true, true, false],
                    vec![true, true, false],
                ],
            },
            graph::GridEdges {
                horizontal: vec![
                    vec![true],
                    vec![false],
                    vec![true],
                ],
                vertical: vec![
                    vec![true, true],
                    vec![true, true],
                ],
            },
        ];
        (piece_id, borders, pieces)
    }

    #[test]
    fn test_curvedata_problem() {
        let (piece_id, borders, pieces) = problem_for_tests();
        let ans = solve_curvedata(&piece_id, &borders, &pieces);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_curvedata_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?curvedata/4/5/=n01o/b0100000/3/3/ec24/2/3/ba1";
        let deserialized = deserialize_problem(url).unwrap();
        assert_eq!(problem, deserialized);
    }
}
