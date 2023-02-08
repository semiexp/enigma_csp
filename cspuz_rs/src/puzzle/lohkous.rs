use super::util;
use crate::graph;
use crate::serializer::strip_prefix;
use crate::solver::Solver;

pub fn solve_lohkous(
    clues: &[Vec<Option<Vec<i32>>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let edges = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&edges.horizontal);
    solver.add_answer_key_bool(&edges.vertical);

    let mut clue_locs = vec![];
    for y in 0..h {
        for x in 0..w {
            if clues[y][x].is_some() {
                clue_locs.push((y, x));
            }
        }
    }

    let block_id = &solver.int_var_2d((h, w), 0, clue_locs.len() as i32 - 1);
    solver.add_expr(
        edges.horizontal.iff(
            block_id
                .slice((..(h - 1), ..))
                .ne(block_id.slice((1.., ..))),
        ),
    );
    solver.add_expr(
        edges.vertical.iff(
            block_id
                .slice((.., ..(w - 1)))
                .ne(block_id.slice((.., 1..))),
        ),
    );

    let max_span = h.max(w);

    for (i, (cy, cx)) in clue_locs.into_iter().enumerate() {
        let clue = clues[cy][cx].as_ref().unwrap();

        solver.add_expr(block_id.at((cy, cx)).eq(i as i32));
        let in_block = &solver.bool_var_2d((h, w));
        solver.add_expr(in_block.iff(block_id.eq(i as i32)));
        graph::active_vertices_connected_2d(&mut solver, in_block);

        let n_right = &solver.int_var_2d((h, w), 0, w as i32);
        solver.add_expr(
            n_right
                .slice_fixed_x((.., w - 1))
                .eq(in_block.slice_fixed_x((.., w - 1)).ite(1, 0)),
        );
        solver.add_expr(
            n_right.slice((.., ..(w - 1))).eq(in_block
                .slice((.., ..(w - 1)))
                .ite(n_right.slice((.., 1..)) + 1, 0)),
        );
        let n_down = &solver.int_var_2d((h, w), 0, h as i32);
        solver.add_expr(
            n_down
                .slice_fixed_y((h - 1, ..))
                .eq(in_block.slice_fixed_y((h - 1, ..)).ite(1, 0)),
        );
        solver.add_expr(
            n_down.slice((..(h - 1), ..)).eq(in_block
                .slice((..(h - 1), ..))
                .ite(n_down.slice((1.., ..)) + 1, 0)),
        );

        let spans = &solver.bool_var_1d(max_span + 1);
        solver.add_expr(!spans.at(0));
        for i in 1..=max_span {
            solver.add_expr(
                spans.at(i).iff(
                    n_right.slice_fixed_x((.., 0)).eq(i as i32).any()
                        | (n_right.slice((.., 1..)).eq(i as i32)
                            & !in_block.slice((.., ..(w - 1))))
                        .any()
                        | n_down.slice_fixed_y((0, ..)).eq(i as i32).any()
                        | (n_down.slice((1.., ..)).eq(i as i32) & !in_block.slice((..(h - 1), ..)))
                            .any(),
                ),
            );
        }

        solver.add_expr(spans.count_true().eq(clue.len() as i32));
        for &c in clue {
            if c > 0 {
                if 1 <= c && c <= max_span as i32 {
                    solver.add_expr(spans.at(c as usize));
                } else {
                    return None;
                }
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(edges))
}

type Problem = Vec<Vec<Option<Vec<i32>>>>;

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let content = strip_prefix(url)?;
    let toks = content.split("/").collect::<Vec<&str>>();
    eprintln!("{:?}", toks);
    if toks[0] != "lohkous" {
        return None;
    }
    let w = toks[1].parse::<usize>().ok()?;
    let h = toks[2].parse::<usize>().ok()?;
    let body = toks[3].as_bytes();
    let mut idx = 0;
    let mut ret = vec![vec![None; w]; h];

    let mut i = 0;
    while i < body.len() {
        if idx >= w * h {
            return None;
        }
        if '0' as u8 <= body[i] && body[i] <= '9' as u8 {
            let mut clues = vec![];
            while i < body.len() {
                if '0' as u8 <= body[i] && body[i] <= '9' as u8 {
                    if body[i] == '0' as u8 {
                        clues.push(-1);
                    } else {
                        clues.push((body[i] - '0' as u8) as i32);
                    }
                } else {
                    break;
                }
                i += 1;
            }
            ret[idx / w][idx % w] = Some(clues);
            idx += 1;
        } else {
            let mut s = (body[i] - 'a' as u8) as usize;
            if i == 0 || i + 1 == body.len() {
                s += 1;
            }
            idx += s;
            i += 1;
        }
    }

    Some(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        // https://puzz.link/p?lohkous/6/6/12k2a23b2k13d13a10b14d
        let mut problem: Vec<Vec<Option<Vec<i32>>>> = vec![vec![None; 6]; 6];
        problem[0][0] = Some(vec![1, 2]);
        problem[1][5] = Some(vec![2]);
        problem[2][0] = Some(vec![2, 3]);
        problem[2][2] = Some(vec![2]);
        problem[4][1] = Some(vec![1, 3]);
        problem[4][5] = Some(vec![1, 3]);
        problem[5][0] = Some(vec![1, -1]);
        problem[5][2] = Some(vec![1, 4]);
        problem
    }

    #[test]
    #[rustfmt::skip]
    fn test_lohkous_problem() {
        let problem = problem_for_tests();

        let ans = solve_lohkous(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1],
                [0, 1, 0, 0, 0],
            ]),
        };
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_lohkous_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?lohkous/6/6/12k2a23b2k13d13a10b14d";
        assert_eq!(deserialize_problem(url).unwrap(), problem);
    }
}
