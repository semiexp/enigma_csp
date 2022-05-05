use super::util;
use crate::graph;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Dict, Grid, HexInt, Optionalize, Spaces,
};
use crate::solver::Solver;

pub fn solve_nurikabe(clues: &[Vec<Option<i32>>]) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);

    let mut clue_pos = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(n) = clues[y][x] {
                clue_pos.push((y, x, n));
            }
        }
    }

    let group_id = solver.int_var_2d((h, w), 0, clue_pos.len() as i32);
    solver.add_expr(is_black.iff(group_id.eq(0)));

    graph::active_vertices_connected_2d(&mut solver, is_black);
    for i in 1..=clue_pos.len() {
        graph::active_vertices_connected_2d(&mut solver, group_id.eq(i as i32));
    }

    solver.add_expr(
        (!(is_black.slice((..(h - 1), ..)) | is_black.slice((1.., ..)))).imp(
            group_id
                .slice((..(h - 1), ..))
                .eq(group_id.slice((1.., ..))),
        ),
    );
    solver.add_expr(
        (!(is_black.slice((.., ..(w - 1))) | is_black.slice((.., 1..)))).imp(
            group_id
                .slice((.., ..(w - 1)))
                .eq(group_id.slice((.., 1..))),
        ),
    );
    solver.add_expr(
        !(is_black.slice((..(h - 1), ..(w - 1)))
            & is_black.slice((..(h - 1), 1..))
            & is_black.slice((1.., ..(w - 1)))
            & is_black.slice((1.., 1..))),
    );

    for (i, &(y, x, n)) in clue_pos.iter().enumerate() {
        solver.add_expr(group_id.at((y, x)).eq((i + 1) as i32));
        if n > 0 {
            solver.add_expr(group_id.eq((i + 1) as i32).count_true().eq(n));
        }
    }

    solver.irrefutable_facts().map(|f| f.get(is_black))
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
    problem_to_url(combinator(), "nurikabe", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["nurikabe"], url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nurikabe_problem() {
        // https://puzz.link/p?nurikabe/6/6/m8n8i9u
        let problem_base = [
            [0, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 8, 0],
            [0, 0, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ];
        let problem = problem_base
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&n| if n == 0 { None } else { Some(n) })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            serialize_problem(&problem),
            Some(String::from("https://puzz.link/p?nurikabe/6/6/m8n8i9u"))
        );
        assert_eq!(
            deserialize_problem("https://puzz.link/p?nurikabe/6/6/m8n8i9u"),
            Some(problem.clone())
        );
        let ans = solve_nurikabe(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_base = [
            [2, 2, 0, 2, 2, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 2, 2],
            [2, 0, 2, 1, 0, 0],
            [2, 0, 2, 0, 2, 2],
            [2, 0, 2, 0, 2, 2],
        ];
        let expected = expected_base.map(|row| {
            row.iter()
                .map(|&n| {
                    if n == 0 {
                        None
                    } else if n == 1 {
                        Some(true)
                    } else {
                        Some(false)
                    }
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(ans, expected);
    }
}
