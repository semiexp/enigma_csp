extern crate cspuz_rs;

use cspuz_rs::puzzle::{nurikabe, yajilin};
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

fn decode_and_solve(url: &[u8]) -> Result<Board, &'static str> {
    let url = std::str::from_utf8(url).map_err(|_| "failed to decode URL as UTF-8")?;

    let puzzle_kind = url_to_puzzle_kind(url).ok_or("puzzle type not detected")?;

    if puzzle_kind == "nurikabe" {
        solve_nurikabe(url)
    } else if puzzle_kind == "yajilin" || puzzle_kind == "yajirin" {
        solve_yajilin(url)
    } else {
        Err("unknown puzzle type")
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
