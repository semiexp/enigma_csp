pub struct Compass {
    pub up: Option<i32>,
    pub down: Option<i32>,
    pub left: Option<i32>,
    pub right: Option<i32>,
}

#[allow(unused)]
pub enum ItemKind {
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
    Compass(Compass),
}

impl ItemKind {
    pub fn to_json(&self) -> String {
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
            ItemKind::Compass(compass) => format!(
                "{{\"kind\":\"compass\",\"up\":{},\"down\":{},\"left\":{},\"right\":{}}}",
                compass.up.unwrap_or(-1),
                compass.down.unwrap_or(-1),
                compass.left.unwrap_or(-1),
                compass.right.unwrap_or(-1)
            ),
        }
    }
}

pub struct Item {
    pub y: usize,
    pub x: usize,
    pub color: &'static str,
    pub kind: ItemKind,
}

impl Item {
    pub fn cell(cell_y: usize, cell_x: usize, color: &'static str, kind: ItemKind) -> Item {
        Item {
            y: cell_y * 2 + 1,
            x: cell_x * 2 + 1,
            color,
            kind,
        }
    }

    pub fn to_json(&self) -> String {
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
pub enum BoardKind {
    Empty,
    Grid,
    OuterGrid,
    DotGrid,
}

pub struct Board {
    pub kind: BoardKind,
    pub height: usize,
    pub width: usize,
    pub data: Vec<Item>,
}

impl Board {
    pub fn to_json(&self) -> String {
        let kind = "grid";
        let height = self.height;
        let width = self.width;
        let default_style = match self.kind {
            BoardKind::Empty => "empty",
            BoardKind::Grid => "grid",
            BoardKind::OuterGrid => "outer_grid",
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
