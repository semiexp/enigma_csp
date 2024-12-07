use crate::graph::InnerGridEdges;
use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, Choice, Combinator, Context, DecInt,
    Dict, KudamonoGrid, KudamonoSequence, Optionalize, PrefixAndSuffix,
};
use crate::solver::{IntVar, IntVarArray1D, Solver};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KropkiClue {
    None,
    White,
    Black,
}

fn add_kropki_constraint(solver: &mut Solver, a: &IntVar, b: &IntVar, clue: KropkiClue) {
    match clue {
        KropkiClue::None => (),
        KropkiClue::White => {
            solver.add_expr(a.eq(b + 1) | a.eq(b - 1));
            solver.add_expr(a.ne(0));
            solver.add_expr(b.ne(0));
        }
        KropkiClue::Black => {
            solver.add_expr(a.eq(b + b) | b.eq(a + a));
            solver.add_expr(a.ne(0));
            solver.add_expr(b.ne(0));
        }
    }
}

pub fn solve_kropki_pairs(
    walls: &InnerGridEdges<Vec<Vec<KropkiClue>>>,
    cells: &[Vec<Option<i32>>],
) -> Option<Vec<Vec<Option<i32>>>> {
    let (h, w) = walls.base_shape();
    let n = h.max(w);

    let mut solver = Solver::new();
    let num = &solver.int_var_2d((h, w), 0, n as i32);
    solver.add_answer_key_int(num);

    for y in 0..h {
        for x in 0..w {
            if let Some(n) = cells[y][x] {
                if n == -1 {
                    solver.add_expr(num.at((y, x)).eq(0));
                } else {
                    solver.add_expr(num.at((y, x)).eq(n));
                }
            } else {
                solver.add_expr(num.at((y, x)).ne(0));
            }
        }
    }

    let mut add_alldifferent = |cells: IntVarArray1D| {
        for i in 0..cells.len() {
            for j in 0..i {
                let x = cells.at(i);
                let y = cells.at(j);
                solver.add_expr(x.eq(0) | y.eq(0) | x.ne(y));
            }
        }
    };

    for y in 0..h {
        add_alldifferent(num.slice_fixed_y((y, ..)));
    }
    for x in 0..w {
        add_alldifferent(num.slice_fixed_x((.., x)));
    }

    for y in 0..h {
        for x in 0..w {
            if y < h - 1 {
                add_kropki_constraint(
                    &mut solver,
                    &num.at((y, x)),
                    &num.at((y + 1, x)),
                    walls.horizontal[y][x],
                );
            }
            if x < w - 1 {
                add_kropki_constraint(
                    &mut solver,
                    &num.at((y, x)),
                    &num.at((y, x + 1)),
                    walls.vertical[y][x],
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(num))
}

type Problem = (InnerGridEdges<Vec<Vec<KropkiClue>>>, Vec<Vec<Option<i32>>>);

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W")?)?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    let cells_combinator = KudamonoGrid::new(
        Choice::new(vec![
            Box::new(Optionalize::new(PrefixAndSuffix::new("(", DecInt, ")"))),
            Box::new(Dict::new(Some(-1), "x")),
        ]),
        None,
    );

    let cells;
    if let Some(p) = parsed.get("L") {
        cells = cells_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        cells = vec![vec![None; width]; height];
    }

    let walls_combinator = KudamonoSequence::new(
        Choice::new(vec![
            Box::new(Dict::new(KropkiClue::White, "w")),
            Box::new(Dict::new(KropkiClue::Black, "b")),
        ]),
        KropkiClue::None,
        height * (width - 1) + width * (height - 1),
    );

    let walls_flat;
    if let Some(p) = parsed.get("L-E") {
        walls_flat = walls_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        walls_flat = vec![KropkiClue::None; height * (width - 1) + width * (height - 1)];
    }

    let mut walls = InnerGridEdges {
        horizontal: vec![vec![KropkiClue::None; width]; height - 1],
        vertical: vec![vec![KropkiClue::None; width - 1]; height],
    };

    for y in 0..(height - 1) {
        for x in 0..width {
            walls.horizontal[y][x] = walls_flat[(height - 2 - y) + x * (2 * height - 1)];
        }
    }
    for y in 0..height {
        for x in 0..(width - 1) {
            walls.vertical[y][x] =
                walls_flat[(height - 1) + (height - 1 - y) + x * (2 * height - 1)];
        }
    }

    Some((walls, cells))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    fn problem_for_tests() -> Problem {
        (InnerGridEdges {
            horizontal: vec![
                vec![KropkiClue::None, KropkiClue::None, KropkiClue::None, KropkiClue::None],
                vec![KropkiClue::White, KropkiClue::None, KropkiClue::Black, KropkiClue::White],
            ],
            vertical: vec![
                vec![KropkiClue::None, KropkiClue::None, KropkiClue::None],
                vec![KropkiClue::None, KropkiClue::None, KropkiClue::None],
                vec![KropkiClue::Black, KropkiClue::Black, KropkiClue::None],
            ],
        }, vec![
            vec![Some(-1), Some(3), None, Some(1)],
            vec![None, None, None, None],
            vec![None, None, None, None],
        ])
    }

    #[test]
    fn test_kropki_pairs_problem() {
        let (walls, cells) = problem_for_tests();
        let ans = solve_kropki_pairs(&walls, &cells);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected =
            crate::puzzle::util::tests::to_option_2d([[0, 3, 4, 1], [3, 1, 2, 4], [4, 2, 1, 3]]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_kropki_pairs_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player.html?W=4x3&L=x2(3)3(1)6&L-E=w0b2b5b3w5&G=kropki-pairs";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
