#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberedArrow {
    Unspecified(i32),
    Up(i32),
    Down(i32),
    Left(i32),
    Right(i32),
}
