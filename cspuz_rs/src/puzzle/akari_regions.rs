use crate::graph;
use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, Choice, Combinator, Context, DecInt,
    Dict, KudamonoBorder, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::{count_true, BoolVar, Solver};

pub fn solve_akari_region(
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
    clues: &[Vec<Option<i32>>], // clue on a cell (not region)
    has_block: &[Vec<bool>],
) -> Option<Vec<Vec<Option<bool>>>> {
    let (h, w) = borders.base_shape();

    let mut solver = Solver::new();
    let has_light = &solver.bool_var_2d((h, w));
    solver.add_answer_key_bool(has_light);

    let mut borders = borders.clone();

    for y in 0..h {
        for x in 0..w {
            if !has_block[y][x] {
                continue;
            }

            solver.add_expr(!has_light.at((y, x)));

            if y > 0 {
                borders.horizontal[y - 1][x] = true;
            }
            if y + 1 < h {
                borders.horizontal[y][x] = true;
            }
            if x > 0 {
                borders.vertical[y][x - 1] = true;
            }
            if x + 1 < w {
                borders.vertical[y][x] = true;
            }
        }
    }

    let rooms = graph::borders_to_rooms(&borders);
    for i in 0..rooms.len() {
        let mut clue: Option<i32> = None;

        for &(y, x) in &rooms[i] {
            if let Some(c) = clues[y][x] {
                if let Some(cc) = clue {
                    if cc != c {
                        return None;
                    }
                } else {
                    clue = Some(c);
                }
            }
        }

        let mut cells = vec![];
        for &pt in &rooms[i] {
            cells.push(has_light.at(pt));
        }

        if let Some(n) = clue {
            solver.add_expr(count_true(cells).eq(n));
        }
    }

    let mut horizontal_group: Vec<Vec<Option<BoolVar>>> = vec![vec![None; w]; h];
    for y in 0..h {
        let mut start: Option<usize> = None;
        for x in 0..=w {
            if x < w && !has_block[y][x] {
                if start.is_none() {
                    start = Some(x);
                }
            } else {
                if let Some(s) = start {
                    let v = solver.bool_var();
                    solver.add_expr(
                        has_light
                            .slice_fixed_y((y, s..x))
                            .count_true()
                            .eq(v.ite(1, 0)),
                    );
                    for x2 in s..x {
                        horizontal_group[y][x2] = Some(v.clone());
                    }
                    println!();
                    start = None;
                }
            }
        }
    }

    let mut vertical_group: Vec<Vec<Option<BoolVar>>> = vec![vec![None; w]; h];
    for x in 0..w {
        let mut start: Option<usize> = None;
        for y in 0..=h {
            if y < h && !has_block[y][x] {
                if start.is_none() {
                    start = Some(y);
                }
            } else {
                if let Some(s) = start {
                    let v = solver.bool_var();
                    solver.add_expr(
                        has_light
                            .slice_fixed_x((s..y, x))
                            .count_true()
                            .eq(v.ite(1, 0)),
                    );
                    for y2 in s..y {
                        vertical_group[y2][x] = Some(v.clone());
                    }
                    start = None;
                }
            }
        }
    }

    for y in 0..h {
        for x in 0..w {
            if !has_block[y][x] {
                solver.add_expr(
                    horizontal_group[y][x].as_ref().unwrap()
                        | vertical_group[y][x].as_ref().unwrap(),
                );
            }
        }
    }

    solver.irrefutable_facts().map(|f| f.get(has_light))
}

pub type Problem = (
    graph::InnerGridEdges<Vec<Vec<bool>>>,
    Vec<Vec<Option<i32>>>,
    Vec<Vec<bool>>,
);

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W")?)?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    let clues;
    if let Some(p) = parsed.get("L-N") {
        let clues_combinator = KudamonoGrid::new(
            Optionalize::new(PrefixAndSuffix::new("(", DecInt, ")")),
            None,
        );
        clues = clues_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        clues = vec![vec![None; width]; height];
    }

    let has_block;
    if let Some(p) = parsed.get("L") {
        let block_combinator =
            KudamonoGrid::new(Choice::new(vec![Box::new(Dict::new(true, "z"))]), false);
        has_block = block_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        has_block = vec![vec![false; width]; height];
    }

    let border;
    if let Some(p) = parsed.get("SIE") {
        border = KudamonoBorder.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        border = graph::InnerGridEdges {
            horizontal: vec![vec![false; width]; height - 1],
            vertical: vec![vec![false; width - 1]; height],
        };
    }

    Some((border, clues, has_block))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        // https://pedros.works/paper-puzzle-player?W=6x5&L=z7z6z8&L-N=(2)3(2)1(1)15(0)4&LF=g2g4g2g4g2g4g7&X=x22x2x2x1x2&SIE=9UL3UU9RURR1U4U5R&G=akari-regional

        let borders = graph::InnerGridEdges {
            horizontal: crate::puzzle::util::tests::to_bool_2d([
                [1, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_bool_2d([
                [0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ]),
        };

        let clues = vec![
            vec![Some(2), None, None, Some(1), None, None],
            vec![Some(2), None, None, None, Some(0), None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
            vec![None, None, None, None, None, None],
        ];

        let has_block = crate::puzzle::util::tests::to_bool_2d([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]);

        (borders, clues, has_block)
    }

    #[test]
    fn test_akari_regions_problem() {
        let problem = problem_for_tests();
        let ans = solve_akari_region(&problem.0, &problem.1, &problem.2);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected = crate::puzzle::util::tests::to_option_bool_2d([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_akari_regions_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=6x5&L=z7z6z8&L-N=(2)3(2)1(1)15(0)4&SIE=9UL3UU9RURR1U4U5R&G=akari-regional";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
