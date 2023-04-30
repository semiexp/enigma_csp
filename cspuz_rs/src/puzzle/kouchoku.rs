use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, AlphaToNum, Choice, Combinator, Context,
    ContextBasedGrid, Dict, Optionalize, Size, Spaces,
};
use crate::solver::{count_true, Solver};

pub type Pt = (usize, usize);

pub fn solve_kouchoku(clues: &[Vec<Option<i32>>]) -> Option<(Vec<(Pt, Pt)>, Vec<(Pt, Pt)>)> {
    let (h, w) = util::infer_shape(clues);

    let mut points = vec![];
    let mut max_num = 0;
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                points.push(((x, y), n));
                max_num = max_num.max(n);
            }
        }
    }

    let mut g = graph::Graph::new(points.len());
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let (pi, ni) = points[i];
            let (pj, nj) = points[j];

            if ni != -1 && nj != -1 && ni != nj {
                continue;
            }
            let mut flg = false;
            for k in 0..points.len() {
                if i != k && j != k && on_line_segment(pi, pj, points[k].0) {
                    flg = true;
                }
            }
            if flg {
                continue;
            }

            g.add_edge(i, j);
        }
    }

    let mut solver = Solver::new();
    let edge_passed = &solver.bool_var_1d(g.n_edges());
    solver.add_answer_key_bool(edge_passed);
    let is_passed = graph::active_edges_single_cycle(&mut solver, edge_passed, &g);
    solver.add_expr(is_passed);

    for i in 0..g.n_edges() {
        for j in 0..i {
            let (p, q) = g[i];
            let (r, s) = g[j];
            if p == r || p == s || q == r || q == s {
                continue;
            }
            if is_cross(points[p].0, points[q].0, points[r].0, points[s].0)
                && !is_perpendicular(points[p].0, points[q].0, points[r].0, points[s].0)
            {
                solver.add_expr(!(edge_passed.at(i) & edge_passed.at(j)));
            }
        }
    }
    let mut boundary = vec![vec![]; (max_num + 1) as usize];
    for i in 0..g.n_edges() {
        let (p, q) = g[i];
        let pn = points[p].1;
        let qn = points[q].1;
        if pn != -1 && qn == -1 {
            boundary[pn as usize].push(edge_passed.at(i));
        } else if pn == -1 && qn != -1 {
            boundary[qn as usize].push(edge_passed.at(i));
        }
    }
    for i in 0..=(max_num as usize) {
        solver.add_expr(count_true(&boundary[i]).eq(2));
    }

    solver.irrefutable_facts().map(|f| {
        let mut fixed_edges = vec![];
        let mut undet_edges = vec![];

        for i in 0..edge_passed.len() {
            let (u, v) = g[i];
            let x = f.get(&edge_passed.at(i));
            if x == Some(true) {
                fixed_edges.push((points[u].0, points[v].0));
            } else if x.is_none() {
                undet_edges.push((points[u].0, points[v].0));
            }
        }

        (fixed_edges, undet_edges)
    })
}

type Problem = Vec<Vec<Option<i32>>>;

fn combinator() -> impl Combinator<Problem> {
    Size::with_offset(
        ContextBasedGrid::new(Choice::new(vec![
            Box::new(Spaces::new_with_maximum(None, '0', '9')),
            Box::new(Optionalize::new(AlphaToNum::new('a', 'z', 0))),
            Box::new(Dict::new(Some(-1), ".")),
        ])),
        1,
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let (h, w) = util::infer_shape(&problem);
    problem_to_url_with_context(
        combinator(),
        "kouchoku",
        problem.clone(),
        &Context::sized(h, w),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["kouchoku"], url)
}

fn to_signed(a: Pt) -> (i64, i64) {
    (a.0 as i64, a.1 as i64)
}

fn signed_area(a: Pt, b: Pt, c: Pt) -> i64 {
    let (ax, ay) = to_signed(a);
    let (bx, by) = to_signed(b);
    let (cx, cy) = to_signed(c);

    (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
}

fn manhattan(a: Pt, b: Pt) -> i64 {
    let (ax, ay) = to_signed(a);
    let (bx, by) = to_signed(b);
    (ax - bx).abs() + (ay - by).abs()
}

/// Returns whether point `p` is on the line segment connecting `a` and `b`.
fn on_line_segment(a: Pt, b: Pt, p: Pt) -> bool {
    if signed_area(a, b, p) != 0 {
        return false;
    }
    manhattan(a, p) + manhattan(p, b) == manhattan(a, b)
}

fn is_cross(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    signed_area(a, b, c).signum() * signed_area(a, b, d).signum() < 0
        && signed_area(c, d, a).signum() * signed_area(c, d, b).signum() < 0
}

fn is_perpendicular(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    let (ax, ay) = to_signed(a);
    let (bx, by) = to_signed(b);
    let (cx, cy) = to_signed(c);
    let (dx, dy) = to_signed(d);

    (bx - ax) * (dx - cx) + (by - ay) * (dy - cy) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        vec![
            vec![Some(1), None, None, Some(-1), None, None, None],
            vec![Some(-1), None, Some(2), None, None, Some(0), None],
            vec![Some(-1), None, None, None, None, None, None],
            vec![None, Some(0), None, Some(2), None, Some(1), None],
            vec![None, None, None, None, None, None, None],
            vec![None, None, None, None, None, None, Some(-1)],
            vec![None, None, None, Some(1), None, None, None],
        ]
    }

    #[test]
    fn test_geometry() {
        assert_eq!(on_line_segment((0, 0), (8, 6), (4, 3)), true);
        assert_eq!(on_line_segment((0, 0), (8, 6), (4, 4)), false);
        assert_eq!(on_line_segment((0, 0), (4, 3), (8, 6)), false);
        assert_eq!(is_cross((0, 0), (1, 1), (0, 1), (1, 0)), true);
        assert_eq!(is_cross((0, 0), (0, 1), (1, 1), (1, 0)), false);
        assert_eq!(is_perpendicular((0, 0), (8, 6), (5, 0), (2, 4)), true);
    }

    #[test]
    fn test_kouchoku_problem() {
        let problem = problem_for_tests();
        let ans = solve_kouchoku(&problem);
        assert!(ans.is_some());
        let mut ans = ans.unwrap();
        ans.0.sort();
        ans.1.sort();

        let expected = (
            vec![
                ((0, 0), (3, 6)),
                ((0, 1), (0, 2)),
                ((2, 1), (3, 3)),
                ((3, 0), (5, 1)),
                ((3, 3), (6, 5)),
                ((5, 1), (1, 3)),
                ((5, 3), (3, 6)),
                ((5, 3), (6, 5)),
            ],
            vec![
                ((0, 0), (0, 1)),
                ((0, 0), (3, 0)),
                ((0, 1), (1, 3)),
                ((0, 2), (1, 3)),
                ((2, 1), (0, 2)),
                ((3, 0), (2, 1)),
            ],
        );
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_kouchoku_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?kouchoku/6/6/b1.2.0c1a0.6a0c0b93.2b2";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
