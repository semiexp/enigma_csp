use cspuz_rs::graph;

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
    FilledCircle,
    SmallCircle,
    SmallFilledCircle,
    SideArrowUp,
    SideArrowDown,
    SideArrowLeft,
    SideArrowRight,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    AboloUpperLeft,
    AboloUpperRight,
    AboloLowerLeft,
    AboloLowerRight,
    Cross,
    Line,
    DoubleLine,
    Wall,
    DottedHorizontalWall,
    DottedVerticalWall,
    BoldWall,
    Text(&'static str),
    Num(i32),
    Compass(Compass),
    TapaClue([i32; 4]),
}

impl ItemKind {
    pub fn to_json(&self) -> String {
        match self {
            &ItemKind::Dot => String::from("\"dot\""),
            &ItemKind::Block => String::from("\"block\""),
            &ItemKind::Fill => String::from("\"fill\""),
            &ItemKind::Circle => String::from("\"circle\""),
            &ItemKind::FilledCircle => String::from("\"filledCircle\""),
            &ItemKind::SmallCircle => String::from("\"smallCircle\""),
            &ItemKind::SmallFilledCircle => String::from("\"smallFilledCircle\""),
            &ItemKind::SideArrowUp => String::from("\"sideArrowUp\""),
            &ItemKind::SideArrowDown => String::from("\"sideArrowDown\""),
            &ItemKind::SideArrowLeft => String::from("\"sideArrowLeft\""),
            &ItemKind::SideArrowRight => String::from("\"sideArrowRight\""),
            &ItemKind::ArrowUp => String::from("\"arrowUp\""),
            &ItemKind::ArrowDown => String::from("\"arrowDown\""),
            &ItemKind::ArrowLeft => String::from("\"arrowLeft\""),
            &ItemKind::ArrowRight => String::from("\"arrowRight\""),
            &ItemKind::AboloUpperLeft => String::from("\"aboloUpperLeft\""),
            &ItemKind::AboloUpperRight => String::from("\"aboloUpperRight\""),
            &ItemKind::AboloLowerLeft => String::from("\"aboloLowerLeft\""),
            &ItemKind::AboloLowerRight => String::from("\"aboloLowerRight\""),
            &ItemKind::Cross => String::from("\"cross\""),
            &ItemKind::Line => String::from("\"line\""),
            &ItemKind::DoubleLine => String::from("\"doubleLine\""),
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
            ItemKind::TapaClue(clues) => format!(
                "{{\"kind\":\"tapaClue\",\"value\":[{},{},{},{}]}}",
                clues[0], clues[1], clues[2], clues[3]
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
    kind: BoardKind,
    height: usize,
    width: usize,
    data: Vec<Item>,
}

impl Board {
    pub fn new(kind: BoardKind, height: usize, width: usize) -> Board {
        Board {
            kind,
            height,
            width,
            data: vec![],
        }
    }

    pub fn push(&mut self, item: Item) {
        self.data.push(item);
    }

    pub fn extend<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = Item>,
    {
        self.data.extend(items);
    }

    pub fn add_borders(&mut self, borders: &graph::BoolInnerGridEdgesModel, color: &'static str) {
        let height = self.height;
        let width = self.width;
        for y in 0..height {
            for x in 0..width {
                if y < height - 1 && borders.horizontal[y][x] {
                    self.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color,
                        kind: ItemKind::BoldWall,
                    });
                }
                if x < width - 1 && borders.vertical[y][x] {
                    self.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color,
                        kind: ItemKind::BoldWall,
                    });
                }
            }
        }
    }

    pub fn add_lines_irrefutable_facts(
        &mut self,
        lines: &graph::BoolGridEdgesIrrefutableFacts,
        color: &'static str,
        skip: Option<&Vec<Vec<bool>>>,
    ) {
        for y in 0..(self.height - 1) {
            for x in 0..self.width {
                if let Some(skip) = skip {
                    if skip[y][x] || skip[y + 1][x] {
                        continue;
                    }
                }
                if let Some(b) = lines.vertical[y][x] {
                    self.push(Item {
                        y: y * 2 + 2,
                        x: x * 2 + 1,
                        color,
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
        }
        for y in 0..self.height {
            for x in 0..(self.width - 1) {
                if let Some(skip) = skip {
                    if skip[y][x] || skip[y][x + 1] {
                        continue;
                    }
                }
                if let Some(b) = lines.horizontal[y][x] {
                    self.push(Item {
                        y: y * 2 + 1,
                        x: x * 2 + 2,
                        color,
                        kind: if b { ItemKind::Line } else { ItemKind::Cross },
                    });
                }
            }
        }
    }

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
