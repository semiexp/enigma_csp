use crate::graph;
use crate::serializer::{
    problem_to_url_with_context, url_to_problem, Choice, Combinator, Context, HexInt, Optionalize,
    RoomsWithValues, Size, Spaces,
};
use crate::solver::{count_true, Solver};

pub fn solve_shimaguni(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Option<i32>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    let rooms = graph::borders_to_rooms(borders);
    assert_eq!(rooms.len(), clues.len());

    let mut idx = vec![vec![(usize::MAX, usize::MAX); w]; h];
    let mut num_black = vec![];
    for i in 0..rooms.len() {
        let room = &rooms[i];
        let mut cells = vec![];
        for j in 0..room.len() {
            cells.push(is_black.at(room[j]));
            idx[room[j].0][room[j].1] = (i, j);
        }
        let n = solver.int_var(0, room.len() as i32);
        solver.add_expr(count_true(&cells).eq(&n));
        solver.add_expr(n.ge(1));
        num_black.push(n);
        let mut graph = graph::Graph::new(room.len());
        for j in 0..room.len() {
            let (y, x) = room[j];
            if y < h - 1 && idx[y + 1][x].0 == i {
                graph.add_edge(j, idx[y + 1][x].1);
            }
            if x < w - 1 && idx[y][x + 1].0 == i {
                graph.add_edge(j, idx[y][x + 1].1);
            }
        }
        graph::active_vertices_connected(&mut solver, &cells, &graph);
    }
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            solver.add_expr(num_black[i].eq(n));
        }
    }

    let mut adj_rooms = vec![];
    for y in 0..h {
        for x in 0..w {
            if y < h - 1 && idx[y][x].0 != idx[y + 1][x].0 {
                let a = idx[y][x].0;
                let b = idx[y + 1][x].0;
                adj_rooms.push((a.min(b), a.max(b)));
                solver.add_expr(!(is_black.at((y, x)) & is_black.at((y + 1, x))));
            }
            if x < w - 1 && idx[y][x].0 != idx[y][x + 1].0 {
                let a = idx[y][x].0;
                let b = idx[y][x + 1].0;
                adj_rooms.push((a.min(b), a.max(b)));
                solver.add_expr(!(is_black.at((y, x)) & is_black.at((y, x + 1))));
            }
        }
    }
    adj_rooms.sort();
    for i in 0..adj_rooms.len() {
        if i == 0 || adj_rooms[i] != adj_rooms[i - 1] {
            let (a, b) = adj_rooms[i];
            solver.add_expr(num_black[a].ne(&num_black[b]));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
}

type Problem = (graph::InnerGridEdges<Vec<Vec<bool>>>, Vec<Option<i32>>);

fn combinator() -> impl Combinator<Problem> {
    Size::new(RoomsWithValues::new(Choice::new(vec![
        Box::new(Optionalize::new(HexInt)),
        Box::new(Spaces::new(None, 'g')),
    ])))
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    let height = problem.0.vertical.len();
    let width = problem.0.vertical[0].len() + 1;
    problem_to_url_with_context(
        combinator(),
        "shimaguni",
        problem.clone(),
        &Context::sized(height, width),
    )
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["shimaguni"], url)
}

#[cfg(test)]
mod tests {
    use super::super::util;
    use super::*;

    fn problem_for_tests() -> Problem {
        let borders = graph::InnerGridEdges {
            horizontal: vec![
                vec![true, true, true, false, false, false],
                vec![true, false, true, true, true, false],
                vec![false, false, true, false, false, false],
                vec![false, true, true, false, false, true],
                vec![false, false, false, false, true, false],
            ],
            vertical: vec![
                vec![false, false, true, true, false],
                vec![true, true, false, true, false],
                vec![true, true, false, true, true],
                vec![true, false, true, true, true],
                vec![false, true, false, true, false],
                vec![false, true, false, false, true],
            ],
        };
        let clues = vec![None, None, None, None, Some(3), None, None, Some(2)];
        (borders, clues)
    }

    #[test]
    fn test_shimaguni_problem() {
        let problem = problem_for_tests();
        let ans = solve_shimaguni(&problem.0, &problem.1);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_base = [
            [0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 1, 0],
        ];
        let expected =
            expected_base.map(|row| row.iter().map(|&n| Some(n == 1)).collect::<Vec<_>>());
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_shimaguni_serializer() {
        let problem = problem_for_tests();
        let url = "https://puzz.link/p?shimaguni/6/6/6qrna9sbh1i2j3h2";
        util::tests::serializer_test(problem, url, serialize_problem, deserialize_problem);
    }
}
