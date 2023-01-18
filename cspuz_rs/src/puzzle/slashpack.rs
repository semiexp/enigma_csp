use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::{count_true, Solver};

pub const SLASHPACK_EMPTY: i32 = 0;
pub const SLASHPACK_SLASH: i32 = 1;
pub const SLASHPACK_BACKSLASH: i32 = 2;

pub fn solve_slashpack(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let ans = &solver.int_var_2d((h, w), 0, 2);
    solver.add_answer_key_int(ans);

    let mut max_num = 0;
    let mut clue_pos = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                max_num = max_num.max(n);
                clue_pos.push((y, x, n));
            }
        }
    }

    let count_num = clue_pos.len() as i32;
    if max_num == 0 || count_num % max_num != 0 {
        return None;
    }

    let n_rooms = count_num / max_num;
    /*
    Each cell contains 4 segments:
    0 /\ 1
    2 \/ 3
     */
    let room_id = &solver.int_var_2d((h * 2, w * 2), 0, n_rooms);
    let room_id_flat = &room_id.flatten();
    let mut g = graph::Graph::new(h * w * 4);
    for y in 0..(h * 2) {
        for x in 0..(w * 2) {
            if y % 2 == x % 2 {
                if y < h * 2 - 1 {
                    if x > 0 {
                        g.add_edge(y * w * 2 + x, (y + 1) * w * 2 + x - 1);
                    }
                    g.add_edge(y * w * 2 + x, (y + 1) * w * 2 + x);
                }
                if x < w * 2 - 1 {
                    g.add_edge(y * w * 2 + x, y * w * 2 + x + 1);
                }
            } else {
                if y < h * 2 - 1 {
                    g.add_edge(y * w * 2 + x, (y + 1) * w * 2 + x);
                }
                if x < w * 2 - 1 {
                    g.add_edge(y * w * 2 + x, y * w * 2 + x + 1);
                }
                if y < h * 2 - 1 && x < w * 2 - 1 {
                    g.add_edge(y * w * 2 + x, (y + 1) * w * 2 + x + 1);
                }
            }
        }
    }
    for i in 0..g.n_edges() {
        let (u, v) = g[i];
        solver.add_expr(
            (room_id_flat.at(u).ne(0) & room_id_flat.at(v).ne(0))
                .imp(room_id_flat.at(u).eq(room_id_flat.at(v))),
        );
    }
    for i in 1..=n_rooms {
        graph::active_vertices_connected(&mut solver, room_id_flat.eq(i), &g);
    }
    for y in 0..h {
        for x in 0..w {
            solver.add_expr(
                room_id
                    .at((y * 2, x * 2))
                    .eq(0)
                    .iff(ans.at((y, x)).eq(SLASHPACK_BACKSLASH)),
            );
            solver.add_expr(
                room_id
                    .at((y * 2, x * 2 + 1))
                    .eq(0)
                    .iff(ans.at((y, x)).eq(SLASHPACK_SLASH)),
            );
            solver.add_expr(
                room_id
                    .at((y * 2 + 1, x * 2))
                    .eq(0)
                    .iff(ans.at((y, x)).eq(SLASHPACK_SLASH)),
            );
            solver.add_expr(
                room_id
                    .at((y * 2 + 1, x * 2 + 1))
                    .eq(0)
                    .iff(ans.at((y, x)).eq(SLASHPACK_BACKSLASH)),
            );
        }
    }
    let num_id = &solver.int_var_1d(clue_pos.len(), 1, n_rooms);
    for (i, &(y, x, _)) in clue_pos.iter().enumerate() {
        solver.add_expr(num_id.at(i).eq(room_id.at((y * 2, x * 2))));
        solver.add_expr(num_id.at(i).eq(room_id.at((y * 2, x * 2 + 1))));
        solver.add_expr(num_id.at(i).eq(room_id.at((y * 2 + 1, x * 2))));
        solver.add_expr(num_id.at(i).eq(room_id.at((y * 2 + 1, x * 2 + 1))));
    }
    for i in 1..=n_rooms {
        solver.add_expr(num_id.eq(i).count_true().eq(max_num));
    }
    for i in 0..clue_pos.len() {
        for j in 0..i {
            if clue_pos[i].2 == clue_pos[j].2 && clue_pos[i].2 > 0 {
                solver.add_expr(num_id.at(i).ne(num_id.at(j)));
            }
        }
    }
    for y in 1..h {
        for x in 1..w {
            solver.add_expr(
                count_true([
                    ans.at((y - 1, x - 1)).eq(SLASHPACK_BACKSLASH),
                    ans.at((y - 1, x)).eq(SLASHPACK_SLASH),
                    ans.at((y, x - 1)).eq(SLASHPACK_SLASH),
                    ans.at((y, x)).eq(SLASHPACK_BACKSLASH),
                ])
                .ne(1),
            );
        }
    }

    solver.irrefutable_facts().map(|f| f.get(ans))
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Grid::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
        Box::new(Dict::new(Some(-1), ".")),
    ]))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "slashpack", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["slashpack"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        vec![
            vec![None, None, Some(1), None, None, None],
            vec![None, Some(1), None, None, None, None],
            vec![None, None, None, Some(1), None, None],
            vec![None, Some(2), None, None, None, None],
            vec![Some(-1), None, None, None, None, None],
            vec![None, None, None, None, Some(-1), None],
        ]
    }

    #[test]
    fn test_slashpack_problem() {
        let problem = problem_for_tests();
        let ans = solve_slashpack(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = vec![
            vec![Some(0), Some(2), Some(0), Some(0), Some(0), Some(0)],
            vec![Some(0), Some(0), Some(2), Some(1), Some(2), Some(0)],
            vec![Some(0), Some(0), Some(1), Some(0), Some(1), Some(0)],
            vec![Some(2), Some(0), Some(2), Some(0), Some(2), Some(0)],
            vec![Some(0), Some(2), Some(1), Some(0), Some(1), Some(0)],
            vec![Some(0), Some(0), Some(0), Some(1), Some(0), Some(0)],
        ];
        assert_eq!(ans, expected);
    }
}
