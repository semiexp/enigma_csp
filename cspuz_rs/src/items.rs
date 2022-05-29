#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberedArrow {
    Unspecified(i32),
    Up(i32),
    Down(i32),
    Left(i32),
    Right(i32),
}

impl NumberedArrow {
    pub fn num(&self) -> i32 {
        match self {
            &NumberedArrow::Unspecified(n) => n,
            &NumberedArrow::Up(n) => n,
            &NumberedArrow::Down(n) => n,
            &NumberedArrow::Left(n) => n,
            &NumberedArrow::Right(n) => n,
        }
    }
}
