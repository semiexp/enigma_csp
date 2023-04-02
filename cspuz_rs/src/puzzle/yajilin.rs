use super::util;
use crate::graph;
use crate::items::NumberedArrow;
use crate::serializer::{
    problem_to_url, url_to_problem, Choice, Combinator, Grid, MaybeSkip, NumberedArrowCombinator,
    Optionalize, Spaces,
};
use crate::solver::Solver;

pub fn solve_yajilin(
    clues: &[Vec<Option<NumberedArrow>>],
) -> Option<(graph::BoolGridEdgesIrrefutableFacts, Vec<Vec<Option<bool>>>)> {
    let (h, w) = util::infer_shape(clues);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let is_passed = &graph::single_cycle_grid_edges(&mut solver, is_line);
    let is_black = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(is_black);
    solver.add_expr(!is_black.conv2d_and((1, 2)));
    solver.add_expr(!is_black.conv2d_and((2, 1)));

    for y in 0..h {
        for x in 0..w {
            if let Some((dir, n)) = clues[y][x] {
                solver.add_expr(!is_passed.at((y, x)));
                solver.add_expr(!is_black.at((y, x)));

                if let Some(cells) = is_black.pointing_cells((y, x), dir) {
                    solver.add_expr(cells.count_true().eq(n));
                }
            } else {
                solver.add_expr(is_passed.at((y, x)) ^ is_black.at((y, x)));
            }
        }
    }

    solver
        .irrefutable_facts()
        .map(|f| (f.get(is_line), f.get(is_black)))
}

type Problem = Vec<Vec<Option<NumberedArrow>>>;

fn combinator() -> impl Combinator<Problem> {
    MaybeSkip::new(
        "b/",
        Grid::new(Choice::new(vec![
            Box::new(Optionalize::new(NumberedArrowCombinator)),
            Box::new(Spaces::new(None, 'a')),
        ])),
    )
}

pub fn serialize_problem(problem: &Problem) -> Option<String> {
    problem_to_url(combinator(), "yajilin", problem.clone())
}

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    url_to_problem(combinator(), &["yajilin", "yajirin"], url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::Arrow;

    #[test]
    fn test_yajilin_problem() {
        // https://puzsq.jp/main/puzzle_play.php?pid=8218
        let mut problem = vec![vec![None; 10]; 10];
        problem[2][3] = Some((Arrow::Left, 2));
        problem[2][5] = Some((Arrow::Right, 1));
        problem[2][8] = Some((Arrow::Down, 1));
        problem[3][0] = Some((Arrow::Down, 1));
        problem[4][3] = Some((Arrow::Down, 2));
        problem[4][9] = Some((Arrow::Left, 0));
        problem[6][3] = Some((Arrow::Down, 1));
        problem[6][5] = Some((Arrow::Up, 2));
        problem[6][8] = Some((Arrow::Up, 1));
        problem[8][7] = Some((Arrow::Down, 0));
        problem[9][2] = Some((Arrow::Left, 0));

        assert_eq!(
            serialize_problem(&problem),
            Some(String::from(
                "https://puzz.link/p?yajilin/10/10/w32a41b21a21l22e30m21a12b11r20d30g"
            ))
        );
        assert_eq!(
            deserialize_problem(
                "https://puzz.link/p?yajilin/10/10/w32a41b21a21l22e30m21a12b11r20d30g"
            ),
            Some(problem.clone())
        );

        let ans = solve_yajilin(&problem);
        assert!(ans.is_some());
        let (_, is_black) = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        ]);
        assert_eq!(is_black, expected);
    }
}
