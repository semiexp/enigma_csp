extern crate cspuz_rs;

use cspuz_rs::graph;
use cspuz_rs::puzzle::{heyawake, nurikabe, slalom, slitherlink, yajilin};
use cspuz_rs::serializer::url_to_puzzle_kind;

static mut SHARED_ARRAY: Vec<u8> = vec![];

#[allow(unused)]
enum ItemKind {
    Dot,
    Block,
    Fill,
    Circle,
    SideArrowUp,
    SideArrowDown,
    SideArrowLeft,
    SideArrowRight,
    Cross,
    Line,
    Wall,
    DottedHorizontalWall,
    DottedVerticalWall,
    BoldWall,
    Text(&'static str),
    Num(i32),
}

impl ItemKind {
    fn to_json(&self) -> String {
        match self {
            &ItemKind::Dot => String::from("\"dot\""),
            &ItemKind::Block => String::from("\"block\""),
            &ItemKind::Fill => String::from("\"fill\""),
            &ItemKind::Circle => String::from("\"circle\""),
            &ItemKind::SideArrowUp => String::from("\"sideArrowUp\""),
            &ItemKind::SideArrowDown => String::from("\"sideArrowDown\""),
            &ItemKind::SideArrowLeft => String::from("\"sideArrowLeft\""),
            &ItemKind::SideArrowRight => String::from("\"sideArrowRight\""),
            &ItemKind::Cross => String::from("\"cross\""),
            &ItemKind::Line => String::from("\"line\""),
            &ItemKind::Wall => String::from("\"wall\""),
            &ItemKind::BoldWall => String::from("\"boldWall\""),
            &ItemKind::DottedHorizontalWall => String::from("\"dottedHorizontalWall\""),
            &ItemKind::DottedVerticalWall => String::from("\"dottedVerticalWall\""),
            &ItemKind::Text(text) => format!("{{\"kind\":\"text\",\"data\":\"{}\"}}", text),
            &ItemKind::Num(num) => format!("{{\"kind\":\"text\",\"data\":\"{}\"}}", num),
        }
    }
}

struct Item {
    y: usize,
    x: usize,
    color: &'static str,
    kind: ItemKind,
}

impl Item {
    fn cell(cell_y: usize, cell_x: usize, color: &'static str, kind: ItemKind) -> Item {
        Item {
            y: cell_y * 2 + 1,
            x: cell_x * 2 + 1,
            color,
            kind,
        }
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"y\":{},\"x\":{},\"color\":\"{}\",\"item\":{}}}",
            self.y,
            self.x,
            self.color,
            self.kind.to_json()
        )
    }
}

#[allow(unused)]
enum BoardKind {
    Empty,
    Grid,
    DotGrid,
}

struct Board {
    kind: BoardKind,
    height: usize,
    width: usize,
    data: Vec<Item>,
}

impl Board {
    fn to_json(&self) -> String {
        let kind = "grid";
        let height = self.height;
        let width = self.width;
        let default_style = match self.kind {
            BoardKind::Empty => "empty",
            BoardKind::Grid => "grid",
            BoardKind::DotGrid => "dots",
        };
        let data = self
            .data
            .iter()
            .map(|item| item.to_json())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"kind\":\"{}\",\"height\":{},\"width\":{},\"defaultStyle\":\"{}\",\"data\":[{}]}}",
            kind, height, width, default_style, data
        )
    }
}

fn solve_nurikabe(url: &str) -> Result<Board, &'static str> {
    let problem = nurikabe::deserialize_problem(url).ok_or("invalid url")?;
    let ans = nurikabe::solve_nurikabe(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut data = vec![];
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                if clue > 0 {
                    data.push(Item::cell(y, x, "black", ItemKind::Num(clue)));
                } else {
                    data.push(Item::cell(y, x, "black", ItemKind::Text("?")));
                }
            } else if let Some(a) = ans[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "green",
                    if a { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }

    Ok(Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    })
}

fn solve_yajilin(url: &str) -> Result<Board, &'static str> {
    use yajilin::YajilinClue;

    let problem = yajilin::deserialize_problem(url).ok_or("invalid url")?;
    let (is_line, is_black) = yajilin::solve_yajilin(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut data = vec![];

    let mut skip_line = vec![];
    for y in 0..height {
        let mut row = vec![];
        for x in 0..width {
            row.push(problem[y][x].is_some() || is_black[y][x] == Some(true));
        }
        skip_line.push(row);
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(clue) = problem[y][x] {
                let (arrow, n) = match clue {
                    YajilinClue::Unspecified(n) => (None, n),
                    YajilinClue::Up(n) => (Some(ItemKind::SideArrowUp), n),
                    YajilinClue::Down(n) => (Some(ItemKind::SideArrowDown), n),
                    YajilinClue::Left(n) => (Some(ItemKind::SideArrowLeft), n),
                    YajilinClue::Right(n) => (Some(ItemKind::SideArrowRight), n),
                };
                if let Some(arrow) = arrow {
                    data.push(Item::cell(y, x, "black", arrow));
                }
                data.push(Item::cell(
                    y,
                    x,
                    "black",
                    if n >= 0 {
                        ItemKind::Num(n)
                    } else {
                        ItemKind::Text("?")
                    },
                ));
            } else if let Some(b) = is_black[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && !(skip_line[y][x] || skip_line[y + 1][x]) {
                if let Some(b) = is_line.vertical[y][x] {
                    data.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
            if x < width - 1 && !(skip_line[y][x] || skip_line[y][x + 1]) {
                if let Some(b) = is_line.horizontal[y][x] {
                    data.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
        }
    }

    Ok(Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    })
}

fn solve_heyawake(url: &str) -> Result<Board, &'static str> {
    let (borders, clues) = heyawake::deserialize_problem(url).ok_or("invalid url")?;
    let is_black = heyawake::solve_heyawake(&borders, &clues).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut data = vec![];

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && borders.horizontal[y][x] {
                data.push(Item {
                    y: y * 2 + 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
            if x < width - 1 && borders.vertical[y][x] {
                data.push(Item {
                    y: y * 2 + 1,
                    x: x * 2 + 2,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "green",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }
    let rooms = graph::borders_to_rooms(&borders);
    assert_eq!(rooms.len(), clues.len());
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let (y, x) = rooms[i][0];
            data.push(Item::cell(y, x, "black", ItemKind::Num(n)));
        }
    }

    Ok(Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    })
}

fn enumerate_answers_heyawake(
    url: &str,
    num_max_answers: usize,
) -> Result<(Board, Vec<Board>), &'static str> {
    let (borders, clues) = heyawake::deserialize_problem(url).ok_or("invalid url")?;
    let is_black_common = heyawake::solve_heyawake(&borders, &clues).ok_or("no answer")?;
    let answers = heyawake::enumerate_answers_heyawake(&borders, &clues, num_max_answers);

    let height = is_black_common.len();
    let width = is_black_common[0].len();

    let mut data = vec![];
    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && borders.horizontal[y][x] {
                data.push(Item {
                    y: y * 2 + 2,
                    x: x * 2 + 1,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
            if x < width - 1 && borders.vertical[y][x] {
                data.push(Item {
                    y: y * 2 + 1,
                    x: x * 2 + 2,
                    color: "black",
                    kind: ItemKind::BoldWall,
                });
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if let Some(b) = is_black_common[y][x] {
                data.push(Item::cell(
                    y,
                    x,
                    "#339933",
                    if b { ItemKind::Block } else { ItemKind::Dot },
                ));
            }
        }
    }
    let rooms = graph::borders_to_rooms(&borders);
    assert_eq!(rooms.len(), clues.len());
    for i in 0..rooms.len() {
        if let Some(n) = clues[i] {
            let (y, x) = rooms[i][0];
            data.push(Item::cell(y, x, "black", ItemKind::Num(n)));
        }
    }

    let board_common = Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    };

    let mut board_answers = vec![];
    for ans in answers {
        let mut data = vec![];
        for y in 0..height {
            for x in 0..width {
                if is_black_common[y][x].is_none() {
                    data.push(Item::cell(
                        y,
                        x,
                        "#cccccc",
                        if ans[y][x] {
                            ItemKind::Block
                        } else {
                            ItemKind::Dot
                        },
                    ));
                }
            }
        }
        board_answers.push(Board {
            kind: BoardKind::Empty,
            height,
            width,
            data,
        });
    }

    Ok((board_common, board_answers))
}

fn solve_slitherlink(url: &str) -> Result<Board, &'static str> {
    let problem = slitherlink::deserialize_problem(url).ok_or("invalid url")?;
    let is_line = slitherlink::solve_slitherlink(&problem).ok_or("no answer")?;

    let height = problem.len();
    let width = problem[0].len();
    let mut data = vec![];

    for y in 0..height {
        for x in 0..width {
            if let Some(n) = problem[y][x] {
                data.push(Item::cell(y, x, "black", ItemKind::Num(n)));
            }
        }
    }
    for y in 0..height {
        for x in 0..=width {
            if let Some(b) = is_line.vertical[y][x] {
                data.push(Item {
                    y: y * 2 + 1,
                    x: x * 2,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }
    for y in 0..=height {
        for x in 0..width {
            if let Some(b) = is_line.horizontal[y][x] {
                data.push(Item {
                    y: y * 2,
                    x: x * 2 + 1,
                    color: "green",
                    kind: if b { ItemKind::Wall } else { ItemKind::Cross },
                })
            }
        }
    }

    Ok(Board {
        kind: BoardKind::DotGrid,
        height,
        width,
        data,
    })
}

fn solve_slalom(url: &str) -> Result<Board, &'static str> {
    use slalom::{SlalomBlackCellDir, SlalomCell};

    let problem = slalom::deserialize_problem_as_primitive(url).ok_or("invalid url")?;
    let (is_black, gates, origin) = slalom::parse_primitive_problem(&problem);
    let is_line = slalom::solve_slalom(origin, &is_black, &gates).ok_or("no answer")?;

    let height = is_black.len();
    let width = is_black[0].len();
    let mut data = vec![];

    let (origin_y, origin_x) = origin;
    data.push(Item::cell(origin_y, origin_x, "black", ItemKind::Circle));
    data.push(Item::cell(
        origin_y,
        origin_x,
        "black",
        ItemKind::Num(gates.len() as i32),
    ));

    for y in 0..height {
        for x in 0..width {
            match problem.0[y][x] {
                SlalomCell::Black(d, n) => {
                    data.push(Item::cell(y, x, "black", ItemKind::Fill));
                    if n >= 0 {
                        data.push(Item::cell(y, x, "white", ItemKind::Num(n)));
                    }
                    let arrow = match d {
                        SlalomBlackCellDir::Up => ItemKind::SideArrowUp,
                        SlalomBlackCellDir::Down => ItemKind::SideArrowDown,
                        SlalomBlackCellDir::Left => ItemKind::SideArrowLeft,
                        SlalomBlackCellDir::Right => ItemKind::SideArrowRight,
                        _ => continue,
                    };
                    data.push(Item::cell(y, x, "white", arrow));
                }
                SlalomCell::Horizontal => {
                    data.push(Item::cell(y, x, "black", ItemKind::DottedHorizontalWall));
                }
                SlalomCell::Vertical => {
                    data.push(Item::cell(y, x, "black", ItemKind::DottedVerticalWall));
                }
                SlalomCell::White => (),
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            if y < height - 1 && !(is_black[y][x] || is_black[y + 1][x]) {
                if let Some(b) = is_line.vertical[y][x] {
                    data.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
            if x < width - 1 && !(is_black[y][x] || is_black[y][x + 1]) {
                if let Some(b) = is_line.horizontal[y][x] {
                    data.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color: "green",
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
        }
    }

    Ok(Board {
        kind: BoardKind::Grid,
        height,
        width,
        data,
    })
}

fn decode_and_solve(url: &[u8]) -> Result<Board, &'static str> {
    let url = std::str::from_utf8(url).map_err(|_| "failed to decode URL as UTF-8")?;

    let puzzle_kind = url_to_puzzle_kind(url).ok_or("puzzle type not detected")?;

    if puzzle_kind == "nurikabe" {
        solve_nurikabe(url)
    } else if puzzle_kind == "yajilin" || puzzle_kind == "yajirin" {
        solve_yajilin(url)
    } else if puzzle_kind == "heyawake" {
        solve_heyawake(url)
    } else if puzzle_kind == "slither" || puzzle_kind == "slitherlink" {
        solve_slitherlink(url)
    } else if puzzle_kind == "slalom" {
        solve_slalom(url)
    } else {
        Err("unknown puzzle type")
    }
}

fn decode_and_enumerate(
    url: &[u8],
    num_max_answers: usize,
) -> Result<(Board, Vec<Board>), &'static str> {
    let url = std::str::from_utf8(url).map_err(|_| "failed to decode URL as UTF-8")?;

    let puzzle_kind = url_to_puzzle_kind(url).ok_or("puzzle type not detected")?;

    if puzzle_kind == "heyawake" {
        enumerate_answers_heyawake(url, num_max_answers)
    } else {
        Err("unsupported puzzle type")
    }
}

#[no_mangle]
fn solve_problem(url: *const u8, len: usize) -> *const u8 {
    let url = unsafe { std::slice::from_raw_parts(url, len) };
    let result = decode_and_solve(url);

    let ret_string = match result {
        Ok(board) => {
            format!("{{\"status\":\"ok\",\"description\":{}}}", board.to_json())
        }
        Err(err) => {
            // TODO: escape `err` if necessary
            format!("{{\"status\":\"error\",\"description\":\"{}\"}}", err)
        }
    };

    let ret_len = ret_string.len();
    unsafe {
        SHARED_ARRAY.clear();
        SHARED_ARRAY.reserve(4 + ret_len);
        SHARED_ARRAY.push((ret_len & 0xff) as u8);
        SHARED_ARRAY.push(((ret_len >> 8) & 0xff) as u8);
        SHARED_ARRAY.push(((ret_len >> 16) & 0xff) as u8);
        SHARED_ARRAY.push(((ret_len >> 24) & 0xff) as u8);
        SHARED_ARRAY.extend_from_slice(ret_string.as_bytes());
        SHARED_ARRAY.as_ptr()
    }
}

#[no_mangle]
fn enumerate_answers_problem(url: *const u8, len: usize, num_max_answers: usize) -> *const u8 {
    let url = unsafe { std::slice::from_raw_parts(url, len) };
    let result = decode_and_enumerate(url, num_max_answers);

    let ret_string = match result {
        Ok((common, per_answer)) => {
            format!(
                "{{\"status\":\"ok\",\"description\":{{\"common\":{},\"answers\":[{}]}}}}",
                common.to_json(),
                per_answer
                    .iter()
                    .map(|x| x.to_json())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Err(err) => {
            // TODO: escape `err` if necessary
            format!("{{\"status\":\"error\",\"description\":\"{}\"}}", err)
        }
    };

    let ret_len = ret_string.len();
    unsafe {
        SHARED_ARRAY.clear();
        SHARED_ARRAY.reserve(4 + ret_len);
        SHARED_ARRAY.push((ret_len & 0xff) as u8);
        SHARED_ARRAY.push(((ret_len >> 8) & 0xff) as u8);
        SHARED_ARRAY.push(((ret_len >> 16) & 0xff) as u8);
        SHARED_ARRAY.push(((ret_len >> 24) & 0xff) as u8);
        SHARED_ARRAY.extend_from_slice(ret_string.as_bytes());
        SHARED_ARRAY.as_ptr()
    }
}
