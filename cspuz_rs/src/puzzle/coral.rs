use crate::graph;
use crate::serializer::{
    problem_to_url_with_context_and_site, url_to_problem, Choice, Combinator, Context, HexInt,
    Optionalize, Seq, Size, Spaces,
};
use crate::solver::{any, count_true, BoolVarArray1D, Solver, TRUE};

pub fn solve_coral(
    clue_vertical: &[Option<Vec<i32>>],
    clue_horizontal: &[Option<Vec<i32>>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let h = clue_horizontal.len();
    let w = clue_vertical.len();

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

    let mut aux_graph = graph::Graph::new(h * w + 1);
    let mut aux_vertices = vec![];
    for y in 0..h {
        for x in 0..w {
            aux_vertices.push(!is_black.at((y, x)));
            if y < h - 1 {
                aux_graph.add_edge(y * w + x, (y + 1) * w + x);
            }
            if x < w - 1 {
                aux_graph.add_edge(y * w + x, y * w + (x + 1));
            }
            if y == 0 || y == h - 1 || x == 0 || x == w - 1 {
                aux_graph.add_edge(y * w + x, h * w);
            }
        }
    }
    aux_vertices.push(TRUE);
    graph::active_vertices_connected(&mut solver, &aux_vertices, &aux_graph);

    for y in 0..h {
        if let Some(clue) = &clue_horizontal[y] {
            if !add_coral_clue(&mut solver, &is_black.slice_fixed_y((y, ..)), clue) {
                return None;
            }
        }
    }
    for x in 0..w {
        if let Some(clue) = &clue_vertical[x] {
            if !add_coral_clue(&mut solver, &is_black.slice_fixed_x((.., x)), clue) {
                return None;
            }
        }
    }
    solver.irrefutable_facts().map(|f| f.get(is_black))
}

type Problem = (Vec<Option<Vec<i32>>>, Vec<Option<Vec<i32>>>);
struct CoralCombinator;

impl Combinator<Problem> for CoralCombinator {
    fn serialize(&self, ctx: &Context, input: &[Problem]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }

        let (clue_vertical, clue_horizontal) = &input[0];
        let h = ctx.height?;
        let w = ctx.width?;
        let mut seq = vec![];
        for i in 0..w {
            let n_filled = if let Some(clue) = &clue_vertical[i] {
                for c in clue {
                    seq.push(Some(*c));
                }
                clue.len()
            } else {
                0
            };
            let n_pad = (h + 1) / 2 - n_filled;
            for _ in 0..n_pad {
                seq.push(None);
            }
        }
        for i in 0..h {
            let n_filled = if let Some(clue) = &clue_horizontal[i] {
                for c in clue {
                    seq.push(Some(*c));
                }
                clue.len()
            } else {
                0
            };
            let n_pad = (w + 1) / 2 - n_filled;
            for _ in 0..n_pad {
                seq.push(None);
            }
        }
        let sub = Seq::new(
            Choice::new(vec![
                Box::new(Optionalize::new(HexInt)),
                Box::new(Spaces::new(None, 'g')),
            ]),
            seq.len(),
        );
        sub.serialize(ctx, &[seq])
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Problem>)> {
        let h = ctx.height?;
        let w = ctx.width?;
        let base_seq_len = ((h + 1) / 2) * w + ((w + 1) / 2) * h;
        let sub = Seq::new(
            Choice::new(vec![
                Box::new(Optionalize::new(HexInt)),
                Box::new(Spaces::new(None, 'g')),
            ]),
            base_seq_len,
        );
        let (n_read, seq) = sub.deserialize(ctx, input)?;
        assert_eq!(seq.len(), 1);
        let seq = &seq[0];
        let mut pos = 0;

        let mut clue_vertical = vec![];
        for _ in 0..w {
            let cs = &seq[pos..(pos + (h + 1) / 2)];
            let cs = cs.iter().filter_map(|&x| x).collect::<Vec<_>>();
            if cs.is_empty() {
                clue_vertical.push(None);
            } else {
                clue_vertical.push(Some(cs));
            }
            pos += (h + 1) / 2;
        }

        let mut clue_horizontal = vec![];
        for _ in 0..h {
            let cs = &seq[pos..(pos + (w + 1) / 2)];
            let cs = cs.iter().filter_map(|&x| x).collect::<Vec<_>>();
            if cs.is_empty() {
                clue_horizontal.push(None);
            } else {
                clue_horizontal.push(Some(cs));
            }
            pos += (w + 1) / 2;
        }

        Some((n_read, vec![(clue_vertical, clue_horizontal)]))
    }
}

fn combinator() -> impl Combinator<Problem> {
    Size::new(CoralCombinator)
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.1.len();
    let width = problem.0.len();
    problem_to_url_with_context_and_site(
        combinator(),
        "coral",
        "https://pzprxs.vercel.app/p?",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["coral"], url)
}

fn add_coral_clue(solver: &mut Solver, cells: &BoolVarArray1D, clue: &Vec<i32>) -> bool {
    let n = cells.len();
    let ord = solver.int_var_1d(n, 0, clue.len() as i32);
    for i in 0..n {
        if i == 0 {
            solver.add_expr(ord.at(i).eq(cells.at(i).ite(1, 0)));
        } else {
            solver.add_expr(
                ord.at(i)
                    .eq(ord.at(i - 1) + (cells.at(i) & !cells.at(i - 1)).ite(1, 0)),
            );
        }
    }
    let mut counts = vec![];
    for i in 0..clue.len() {
        let c = solver.int_var(1, cells.len() as i32);
        solver.add_expr((ord.eq(i as i32 + 1) & cells).count_true().eq(&c));
        counts.push(c);
    }
    let mut bucket = vec![0; n + 1];
    for &c in clue {
        if !(1 <= c && c <= n as i32) {
            return false;
        }
        bucket[c as usize] += 1;
    }
    for i in 0..clue.len() {
        let mut cand = vec![];
        for j in 1..=n {
            if bucket[j] > 0 {
                cand.push(counts[i].eq(j as i32));
            }
        }
        solver.add_expr(any(cand));
    }
    for j in 1..=n {
        if bucket[j] == 0 {
            continue;
        }
        let mut cand = vec![];
        for i in 0..clue.len() {
            cand.push(counts[i].eq(j as i32));
        }
        solver.add_expr(count_true(cand).eq(bucket[j]));
    }

    true
}

#[cfg(test)]
mod tests {
    use super::super::util;
    use super::*;

    fn problem_for_tests() -> (Vec<Option<Vec<i32>>>, Vec<Option<Vec<i32>>>) {
        let clue_vertical = vec![
            Some(vec![1]),
            Some(vec![3, 1, 1]),
            Some(vec![3, 3]),
            None,
            None,
            Some(vec![1, 1, 1]),
        ];
        let clue_horizontal = vec![
            None,
            Some(vec![2, 1]),
            Some(vec![2, 1]),
            Some(vec![2, 1]),
            Some(vec![3]),
            None,
            Some(vec![2]),
        ];
        (clue_vertical, clue_horizontal)
    }

    #[test]
    fn test_coral_problem() {
        let problem = problem_for_tests();
        let ans = solve_coral(&problem.0, &problem.1);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_base = [
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0],
        ];
        let expected =
            expected_base.map(|row| row.iter().map(|&n| Some(n == 1)).collect::<Vec<_>>());
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_coral_serializer() {
        let problem = problem_for_tests();
        let url = "https://pzprxs.vercel.app/p?coral/6/7/1i311g33p111j21g21g21g3k2h";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
