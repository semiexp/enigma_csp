extern crate cspuz_rs;

use cspuz_rs::puzzle::nurikabe;

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

fn decode_and_solve(url: &[u8]) -> Result<Board, &'static str> {
    let url = std::str::from_utf8(url).map_err(|_| "failed to decode URL as UTF-8")?;

    // TODO: support other puzzle kinds
    solve_nurikabe(url)
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
