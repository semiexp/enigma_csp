use super::util;
use crate::graph;
use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, Choice, Combinator, Context, DecInt,
    Dict, KudamonoBorder, KudamonoGrid, Optionalize, PrefixAndSuffix,
};
use crate::solver::{count_true, Solver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CBPLCell {
    Empty,
    Blocked,
    BlackCircle,
    WhiteCircle,
}

pub fn solve_cross_border_parity_loop(
    board: &[Vec<CBPLCell>],
    clues_black: &[Vec<Option<i32>>],
    clues_white: &[Vec<Option<i32>>],
    borders: &graph::InnerGridEdges<Vec<Vec<bool>>>,
) -> Option<(graph::BoolGridEdgesIrrefutableFacts, Vec<Vec<Option<i32>>>)> {
    let (h, w) = util::infer_shape(board);

    let mut solver = Solver::new();
    let is_line = &graph::BoolGridEdges::new(&mut solver, (h - 1, w - 1));
    solver.add_answer_key_bool(&is_line.horizontal);
    solver.add_answer_key_bool(&is_line.vertical);

    let color = &solver.bool_var_2d((h, w));
    let is_passed = &graph::single_cycle_grid_edges(&mut solver, &is_line);
    solver.add_expr(is_passed.any());

    let pass_type = &solver.int_var_2d((h, w), 0, 2);
    solver.add_answer_key_int(pass_type);
    solver.add_expr(pass_type.eq(0).iff(!is_passed));
    solver.add_expr(pass_type.eq(1).iff(is_passed & color));
    solver.add_expr(pass_type.eq(2).iff(is_passed & !color));

    for y in 0..h {
        for x in 0..w {
            match board[y][x] {
                CBPLCell::Empty => (),
                CBPLCell::Blocked => solver.add_expr(!is_passed.at((y, x))),
                CBPLCell::BlackCircle => {
                    solver.add_expr(is_passed.at((y, x)).imp(color.at((y, x))))
                }
                CBPLCell::WhiteCircle => {
                    solver.add_expr(is_passed.at((y, x)).imp(!color.at((y, x))))
                }
            }
            if y < h - 1 {
                solver.add_expr(
                    is_line
                        .vertical
                        .at((y, x))
                        .imp(color.at((y, x)) ^ color.at((y + 1, x)) ^ !borders.horizontal[y][x]),
                );
            }
            if x < w - 1 {
                solver.add_expr(
                    is_line
                        .horizontal
                        .at((y, x))
                        .imp(color.at((y, x)) ^ color.at((y, x + 1)) ^ !borders.vertical[y][x]),
                );
            }
        }
    }

    let rooms = graph::borders_to_rooms(borders);
    for room in rooms {
        for &(y, x) in &room {
            if let Some(n) = clues_black[y][x] {
                let mut constr = vec![];
                for &p in &room {
                    constr.push(is_passed.at(p) & color.at(p));
                }
                if n >= 0 {
                    solver.add_expr(count_true(constr).eq(n));
                }
            }
            if let Some(n) = clues_white[y][x] {
                let mut constr = vec![];
                for &p in &room {
                    constr.push(is_passed.at(p) & !color.at(p));
                }
                if n >= 0 {
                    solver.add_expr(count_true(constr).eq(n));
                }
            }
        }
    }

    solver
        .irrefutable_facts()
        .map(|f| (f.get(is_line), f.get(pass_type)))
}

type Problem = (
    Vec<Vec<CBPLCell>>,
    Vec<Vec<Option<i32>>>,
    Vec<Vec<Option<i32>>>,
    graph::InnerGridEdges<Vec<Vec<bool>>>,
);

pub fn deserialize_problem(url: &str) -> Option<Problem> {
    let parsed = get_kudamono_url_info_detailed(url)?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W")?)?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    let bw_combinator = KudamonoGrid::new(
        Optionalize::new(PrefixAndSuffix::new("(", DecInt, ")")),
        None,
    );

    let clues_black;
    if let Some(p) = parsed.get("LI-S") {
        clues_black = bw_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        clues_black = vec![vec![None; width]; height];
    }

    let clues_white;
    if let Some(p) = parsed.get("LI-N") {
        clues_white = bw_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        clues_white = vec![vec![None; width]; height];
    }

    let board;
    if let Some(p) = parsed.get("L") {
        let board_combinator = KudamonoGrid::new(
            Choice::new(vec![
                Box::new(Dict::new(CBPLCell::Blocked, "x")),
                Box::new(Dict::new(CBPLCell::BlackCircle, "b")),
                Box::new(Dict::new(CBPLCell::WhiteCircle, "w")),
            ]),
            CBPLCell::Empty,
        );
        board = board_combinator.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        board = vec![vec![CBPLCell::Empty; width]; height];
    }

    let mut border;
    if let Some(p) = parsed.get("SIE") {
        border = KudamonoBorder.deserialize(&ctx, p.as_bytes())?.1.pop()?;
    } else {
        border = graph::InnerGridEdges {
            horizontal: vec![vec![false; width]; height - 1],
            vertical: vec![vec![false; width - 1]; height],
        };
    }

    for y in 0..height {
        for x in 0..width {
            if board[y][x] == CBPLCell::Blocked {
                if y > 0 {
                    border.horizontal[y - 1][x] = true;
                }
                if y < height - 1 {
                    border.horizontal[y][x] = true;
                }
                if x > 0 {
                    border.vertical[y][x - 1] = true;
                }
                if x < width - 1 {
                    border.vertical[y][x] = true;
                }
            }
        }
    }

    Some((board, clues_black, clues_white, border))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        let height = 5;
        let width = 6;

        let mut board = vec![vec![CBPLCell::Empty; width]; height];
        board[0][0] = CBPLCell::Blocked;
        board[0][2] = CBPLCell::WhiteCircle;
        board[1][1] = CBPLCell::WhiteCircle;
        board[1][4] = CBPLCell::WhiteCircle;
        board[2][2] = CBPLCell::Blocked;
        board[3][5] = CBPLCell::BlackCircle;
        board[4][4] = CBPLCell::BlackCircle;

        let mut clues_black = vec![vec![None; width]; height];
        clues_black[2][0] = Some(2);
        clues_black[3][3] = Some(1);

        let mut clues_white = vec![vec![None; width]; height];
        clues_white[1][1] = Some(1);
        clues_white[3][3] = Some(3);

        let borders = graph::InnerGridEdges {
            horizontal: crate::puzzle::util::tests::to_bool_2d([
                [1, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ]),
            vertical: crate::puzzle::util::tests::to_bool_2d([
                [1, 1, 0, 1, 1],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
            ]),
        };

        (board, clues_black, clues_white, borders)
    }

    #[test]
    fn test_cbpl_problem() {
        let (board, clues_black, clues_white, borders) = problem_for_tests();
        let ans = solve_cross_border_parity_loop(&board, &clues_black, &clues_white, &borders);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected_loop = graph::BoolGridEdgesIrrefutableFacts {
            horizontal: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0],
            ]),
            vertical: crate::puzzle::util::tests::to_option_bool_2d([
                [0, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0],
            ]),
        };

        let expected_cell = crate::puzzle::util::tests::to_option_2d([
            [0, 0, 0, 1, 2, 1],
            [0, 2, 1, 1, 0, 1],
            [0, 1, 0, 2, 1, 1],
            [0, 1, 2, 1, 2, 0],
            [0, 0, 2, 2, 1, 0],
        ]);

        assert_eq!(ans, (expected_loop, expected_cell));
    }

    #[test]
    fn test_cbpl_serializer() {
        let problem = problem_for_tests();
        let url = "https://pedros.works/paper-puzzle-player?W=6x5&LI-N=(1)8(3)8&LI-S=(2)2(1)14&L=x4w4x4w2b6w3b3&SIE=3RRUU9UU8RRR4UUUU1RR10DLU&G=cross-border-parity-loop";
        assert_eq!(deserialize_problem(url), Some(problem));
    }
}
