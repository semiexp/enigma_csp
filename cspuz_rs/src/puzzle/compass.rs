use crate::graph;
use crate::solver::Solver;

#[derive(Clone, Copy)]
pub struct CompassClue {
    up: Option<i32>,
    down: Option<i32>,
    left: Option<i32>,
    right: Option<i32>,
}

pub fn solve_compass(
    clues: &[Vec<Option<CompassClue>>],
) -> Option<graph::BoolInnerGridEdgesIrrefutableFacts> {
    let h = clues.len();
    assert!(h > 0);
    let w = clues[0].len();

    let mut solver = Solver::new();
    let edges = &graph::BoolInnerGridEdges::new(&mut solver, (h, w));
    solver.add_answer_key_bool(&edges.horizontal);
    solver.add_answer_key_bool(&edges.vertical);

    let mut compasses = vec![];
    for y in 0..h {
        for x in 0..w {
            if let Some(c) = clues[y][x] {
                compasses.push((y, x, c));
            }
        }
    }
    let group_id = solver.int_var_2d((h, w), 0, compasses.len() as i32 - 1);
    solver.add_expr(
        edges.horizontal.iff(
            group_id
                .slice((..(h - 1), ..))
                .ne(group_id.slice((1.., ..))),
        ),
    );
    solver.add_expr(
        edges.vertical.iff(
            group_id
                .slice((.., ..(w - 1)))
                .ne(group_id.slice((.., 1..))),
        ),
    );
    for (i, &(y, x, c)) in compasses.iter().enumerate() {
        graph::active_vertices_connected_2d(&mut solver, group_id.eq(i as i32));
        solver.add_expr(group_id.at((y, x)).eq(i as i32));
        if let Some(n) = c.up {
            solver.add_expr(group_id.slice((..y, ..)).eq(i as i32).count_true().eq(n));
        }
        if let Some(n) = c.down {
            solver.add_expr(
                group_id
                    .slice(((y + 1).., ..))
                    .eq(i as i32)
                    .count_true()
                    .eq(n),
            );
        }
        if let Some(n) = c.left {
            solver.add_expr(group_id.slice((.., ..x)).eq(i as i32).count_true().eq(n));
        }
        if let Some(n) = c.right {
            solver.add_expr(
                group_id
                    .slice((.., (x + 1)..))
                    .eq(i as i32)
                    .count_true()
                    .eq(n),
            );
        }
    }

    solver.irrefutable_facts().map(|f| f.get(edges))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_compass_problem() {
        // https://puzz.link/p?compass/5/5/m..1.i25.1g53..i1..1m
        let mut problem: Vec<Vec<Option<CompassClue>>> = vec![vec![None; 5]; 5];
        problem[1][2] = Some(CompassClue {
            up: None,
            down: None,
            left: Some(1),
            right: None,
        });
        problem[2][1] = Some(CompassClue {
            up: Some(2),
            down: Some(5),
            left: None,
            right: Some(1),
        });
        problem[2][3] = Some(CompassClue {
            up: Some(5),
            down: Some(3),
            left: None,
            right: None,
        });
        problem[3][2] = Some(CompassClue {
            up: Some(1),
            down: None,
            left: None,
            right: Some(1),
        });

        let ans = solve_compass(&problem);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = graph::BoolInnerGridEdgesIrrefutableFacts {
            horizontal: vec![
                vec![Some(false), Some(true), Some(true), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(true), Some(true), Some(false)],
                vec![Some(false), Some(false), Some(false), Some(true), Some(false)],
                vec![Some(false), Some(false), Some(true), Some(true), Some(false)],
            ],
            vertical: vec![
                vec![Some(true), Some(false), Some(false), Some(false)],
                vec![Some(true), Some(false), Some(false), Some(true)],
                vec![Some(false), Some(true), Some(true), Some(false)],
                vec![Some(false), Some(true), Some(false), Some(true)],
                vec![Some(false), Some(false), Some(true), Some(false)],
            ],
        };
        assert_eq!(ans, expected);
    }
}
