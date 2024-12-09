use crate::serializer::{
    get_kudamono_url_info_detailed, parse_kudamono_dimension, AlphaToNum, Choice, Combinator,
    Context, DecInt, Dict, KudamonoGrid, KudamonoSequence, Map, PrefixAndSuffix,
};
use crate::solver::{int_constant, Solver};

pub fn solve_letter_weights(
    sums: &[(Vec<char>, i32)],
    chars: &[char],
    nums: &[i32],
) -> (Vec<char>, Vec<i32>, Option<Vec<Vec<Option<bool>>>>) {
    assert_eq!(chars.len(), nums.len());

    let mut nums = nums.to_vec();
    nums.sort();

    let mut nums_with_count = vec![];
    let mut current = None;
    let mut count = 0;
    for &num in &nums {
        if current == Some(num) {
            count += 1;
        } else {
            if let Some(current) = current {
                nums_with_count.push((current, count));
            }
            current = Some(num);
            count = 1;
        }
    }
    if let Some(current) = current {
        nums_with_count.push((current, count));
    }

    let unique_nums = nums_with_count
        .iter()
        .map(|(num, _)| *num)
        .collect::<Vec<_>>();

    let mut solver = Solver::new();
    let mapping = &solver.bool_var_2d((chars.len(), unique_nums.len()));
    solver.add_answer_key_bool(mapping);

    let mut values = vec![];

    for i in 0..chars.len() {
        values.push(solver.int_var_from_domain(unique_nums.clone()));

        for j in 0..unique_nums.len() {
            solver.add_expr(mapping.at((i, j)).iff(values[i].eq(unique_nums[j])));
        }
    }

    for i in 0..nums_with_count.len() {
        solver.add_expr(
            mapping
                .slice_fixed_x((.., i))
                .count_true()
                .eq(nums_with_count[i].1),
        );
    }

    for i in 0..sums.len() {
        let mut e = int_constant(0);

        for &c in &sums[i].0 {
            let idx = chars.iter().position(|&x| x == c).unwrap();
            e = e + &values[idx];
        }

        solver.add_expr(e.eq(sums[i].1));
    }

    let ans = solver.irrefutable_facts().map(|f| f.get(mapping));

    (chars.to_vec(), unique_nums, ans)
}

type Problem = (Vec<(Vec<char>, i32)>, Vec<char>, Vec<i32>);

#[derive(Clone, Debug, PartialEq, Eq)]
enum CellValue {
    Num(i32),
    Alpha(i32),
    Empty,
}

pub fn deserialize_problem(url: &str) -> Result<Problem, &'static str> {
    let parsed = get_kudamono_url_info_detailed(url).ok_or("invalid url")?;
    let (width, height) = parse_kudamono_dimension(parsed.get("W").ok_or("dimension not found")?)
        .ok_or("dimension could not be parsed")?;

    let ctx = Context::sized_with_kudamono_mode(height, width, true);

    let cells_combinator = KudamonoGrid::new(
        Choice::new(vec![
            Box::new(Map::new(
                PrefixAndSuffix::new("(", DecInt, ")"),
                |_| None,
                |v| Some(CellValue::Num(v)),
            )),
            Box::new(Map::new(
                AlphaToNum::new('A', 'Z', 0),
                |_| None,
                |v| Some(CellValue::Alpha(v)),
            )),
            Box::new(Dict::new(CellValue::Empty, "x")),
        ]),
        CellValue::Alpha(-1),
    );

    let cells = cells_combinator
        .deserialize(&ctx, parsed.get("L").ok_or("cells not found")?.as_bytes())
        .ok_or("cells could not be parsed")?
        .1
        .pop()
        .unwrap();

    let mut nonempty_cols = vec![];
    for x in 0..width {
        let mut nonempty = false;
        for y in 0..height {
            if cells[y][x] != CellValue::Empty {
                nonempty = true;
                break;
            }
        }
        if nonempty {
            nonempty_cols.push(x);
        }
    }

    if nonempty_cols.len() < 3 {
        return Err("too few nonempty columns");
    }

    let chars_col = nonempty_cols[nonempty_cols.len() - 1];
    let nums_col = nonempty_cols[nonempty_cols.len() - 2];
    let last_cell_col = nonempty_cols[nonempty_cols.len() - 3];

    if chars_col - nums_col == 1 {
        return Err("invalid column configuration");
    }
    if nums_col - last_cell_col == 1 {
        return Err("invalid column configuration");
    }

    let walls_combinator = KudamonoSequence::new(
        Choice::new(vec![
            Box::new(Dict::new(Some('+'), "p")),
            Box::new(Dict::new(Some('='), "e")),
        ]),
        None,
        height * (width - 1) + width * (height - 1),
    );

    let mut nums = vec![];
    for y in 0..height {
        match cells[y][nums_col] {
            CellValue::Num(n) => nums.push(n),
            CellValue::Empty => (),
            _ => return Err("alphabet cannot exist in nums column"),
        }
    }

    let mut chars = vec![];
    for y in 0..height {
        match cells[y][chars_col] {
            CellValue::Alpha(n) => chars.push((n + 'A' as i32) as u8 as char),
            CellValue::Empty => (),
            _ => return Err("numbers cannot exist in chars column"),
        }
    }

    let ops_flat = walls_combinator
        .deserialize(
            &ctx,
            parsed
                .get("L-MATH")
                .ok_or("operators not found")?
                .as_bytes(),
        )
        .ok_or("operators could not be parsed")?
        .1
        .pop()
        .unwrap();
    let mut ops = vec![vec![None; width - 1]; height];

    for i in 0..ops_flat.len() {
        if let Some(c) = ops_flat[i] {
            let x = i / (2 * height - 1);
            let yb = i % (2 * height - 1);

            if yb < height - 1 || x >= last_cell_col {
                return Err("invalid operator position");
            }

            let y = (2 * height - 2) - yb;
            ops[y][x] = Some(c);
        }
    }

    let mut sums = vec![];

    for y in 0..height {
        let mut state = 0;
        let mut cur_sum = vec![];

        for x in 0..=last_cell_col {
            if state == 0 {
                match cells[y][x] {
                    CellValue::Alpha(n) => {
                        state = 1;
                        cur_sum.push((n + 'A' as i32) as u8 as char);
                    }
                    CellValue::Empty => (),
                    CellValue::Num(_) => return Err("invalid cell values"),
                }
            } else if state == 1 {
                match cells[y][x] {
                    CellValue::Alpha(n) => {
                        cur_sum.push((n + 'A' as i32) as u8 as char);
                    }
                    _ => return Err("invalid cell values"),
                }
            } else if state == 2 {
                match cells[y][x] {
                    CellValue::Num(n) => {
                        state = 0;
                        sums.push((cur_sum, n));
                        cur_sum = vec![];
                    }
                    _ => return Err("invalid cell values"),
                }
            }

            if state == 0 {
                if ops[y][x].is_some() {
                    return Err("invalid operator position");
                }
            } else if state == 1 {
                if ops[y][x] == Some('+') {
                    // do nothing
                } else if ops[y][x] == Some('=') {
                    state = 2;
                } else {
                    return Err("invalid operator position");
                }
            } else if state == 2 {
                return Err("invalid cell values");
            }
        }
    }

    Ok((sums, chars, nums))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem_for_tests() -> Problem {
        (
            vec![(vec!['D', 'O', 'O', 'R'], 18), (vec!['D', 'O'], 4)],
            vec!['D', 'O', 'R'],
            vec![1, 3, 11],
        )
    }

    #[test]
    fn test_letter_weights_problem() {
        let (sums, chars, nums) = problem_for_tests();
        let (_, _, ans) = solve_letter_weights(&sums, &chars, &nums);
        assert!(ans.is_some());
        let ans = ans.unwrap();

        let expected =
            crate::puzzle::util::tests::to_option_bool_2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        assert_eq!(ans, expected);
    }

    #[test]
    fn test_letter_weights_serializer() {
        {
            let url = "https://pedros.works/paper-puzzle-player?W=9x5&L=x0x1D1D1x1x1x1O1O1x1x1x1(4)1O1x1x1x1x1R1x1x1x1x1(18)1x1x1x1x1x1x1x1(11)1(3)1(1)1x1x1x1x1x1x1x1R1O1D1x1&L-MATH=p6p1e8p1p9e9&G=letter-weights";
            let problem = problem_for_tests();
            assert_eq!(deserialize_problem(url), Ok(problem));
        }

        {
            let url = "https://pedros.works/paper-puzzle-player?W=9x5&L=x0x1D1D1x1x1x1O1O1x1x1x1(4)1O1x1x1x1x1R1x1x1x1x1(18)1x1x1x1x1x1x1x1(11)1(3)1(1)1x1x1x1x1x1x1x1R1O1D1x1&L-MATH=p6p1e8p1e18&G=letter-weights";
            assert_eq!(deserialize_problem(url), Err("invalid operator position"));
        }

        {
            let url = "https://pedros.works/paper-puzzle-player?W=9x5&L=x0x1D1D1x1x1x1O1O1x1x1x1(4)1O1x1x1x1x1R1x1x1x1x1(18)1x1x1x1x1x1x1A1(11)1(3)1(1)1x1x1x1x1x1x1x1R1O1D1x1&L-MATH=p6p1e8p1e18&G=letter-weights";
            assert_eq!(
                deserialize_problem(url),
                Err("alphabet cannot exist in nums column")
            );
        }
    }
}
