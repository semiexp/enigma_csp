#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrow {
    Unspecified,
    Up,
    Down,
    Left,
    Right,
}

pub type NumberedArrow = (Arrow, i32);
