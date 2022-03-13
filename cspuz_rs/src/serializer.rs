use std::ops::Index;

fn is_dec(c: u8) -> bool {
    return '0' as u8 <= c && c <= '9' as u8;
}

fn is_hex(c: u8) -> bool {
    return ('0' as u8 <= c && c <= '9' as u8) || ('a' as u8 <= c && c <= 'f' as u8);
}

fn is_base36(c: u8) -> bool {
    return ('0' as u8 <= c && c <= '9' as u8) || ('a' as u8 <= c && c <= 'z' as u8);
}

fn to_base36(n: i32) -> u8 {
    assert!(0 <= n && n < 36);
    if n <= 9 {
        n as u8 + '0' as u8
    } else {
        (n - 10) as u8 + 'a' as u8
    }
}

fn from_base36(c: u8) -> i32 {
    assert!(is_base36(c));
    if '0' as u8 <= c && c <= '9' as u8 {
        (c - '0' as u8) as i32
    } else {
        (c - 'a' as u8) as i32 + 10
    }
}

fn to_base16(n: i32) -> u8 {
    assert!(0 <= n && n < 16);
    to_base36(n)
}

fn from_base16(c: u8) -> i32 {
    assert!(is_hex(c));
    from_base36(c)
}

pub trait Combinator<T> {
    fn serialize(&self, input: &[T]) -> Option<(usize, Vec<u8>)>;
    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<T>)>;
}

impl<A, T> Combinator<T> for &A
where
    A: Combinator<T>,
{
    fn serialize(&self, input: &[T]) -> Option<(usize, Vec<u8>)> {
        (*self).serialize(input)
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<T>)> {
        (*self).deserialize(input)
    }
}

pub struct Choice<T> {
    choices: Vec<Box<dyn Combinator<T>>>,
}

impl<T> Choice<T> {
    pub fn new(choices: Vec<Box<dyn Combinator<T>>>) -> Choice<T> {
        Choice { choices }
    }
}

impl<T> Combinator<T> for Choice<T> {
    fn serialize(&self, input: &[T]) -> Option<(usize, Vec<u8>)> {
        self.choices
            .iter()
            .find_map(|choice| choice.serialize(input))
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<T>)> {
        self.choices
            .iter()
            .find_map(|choice| choice.deserialize(input))
    }
}

pub struct Spaces<T: Clone + PartialEq> {
    space: T,
    minimum: u8,
    maximum: u8,
}

impl<T: Clone + PartialEq> Spaces<T> {
    pub fn new(space: T, minimum: char) -> Spaces<T> {
        Spaces {
            space,
            minimum: minimum as u8,
            maximum: 'z' as u8,
        }
    }
}

impl<T: Clone + PartialEq> Combinator<T> for Spaces<T> {
    fn serialize(&self, input: &[T]) -> Option<(usize, Vec<u8>)> {
        let n_spaces_max = (self.maximum - self.minimum) as usize + 1;
        let mut n_spaces = 0;
        while n_spaces < input.len() && n_spaces < n_spaces_max && input[n_spaces] == self.space {
            n_spaces += 1;
        }
        if n_spaces == 0 {
            None
        } else {
            Some((n_spaces, vec![self.minimum + (n_spaces - 1) as u8]))
        }
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<T>)> {
        if input.len() == 0 {
            return None;
        }
        let v = input[0];
        if !(self.minimum <= v && v <= self.maximum) {
            return None;
        }
        let mut ret = vec![];
        for _ in 0..=(v - self.minimum) {
            ret.push(self.space.clone());
        }
        Some((1, ret))
    }
}

pub struct HexInt;

impl Combinator<i32> for HexInt {
    fn serialize(&self, input: &[i32]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let v = input[0];
        if 0 <= v && v < 16 {
            Some((1, vec![to_base16(v)]))
        } else if 16 <= v && v < 256 {
            Some((1, vec!['-' as u8, to_base16(v >> 4), to_base16(v & 15)]))
        } else if 256 <= v && v < 4096 {
            Some((
                1,
                vec![
                    '+' as u8,
                    to_base16(v >> 8),
                    to_base16((v >> 4) & 15),
                    to_base16(v & 15),
                ],
            ))
        } else {
            None
        }
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<i32>)> {
        if input.len() == 0 {
            return None;
        }
        let v = input[0];
        if is_hex(v) {
            Some((1, vec![from_base16(v)]))
        } else if v == '-' as u8 {
            if !(input.len() >= 3 && is_hex(input[1]) && is_hex(input[2])) {
                None
            } else {
                Some((
                    3,
                    vec![(from_base16(input[1]) << 4) | from_base16(input[2])],
                ))
            }
        } else if v == '+' as u8 {
            if !(input.len() >= 4 && is_hex(input[1]) && is_hex(input[2]) && is_hex(input[3])) {
                None
            } else {
                Some((
                    4,
                    vec![
                        (from_base16(input[1]) << 8)
                            | (from_base16(input[2]) << 4)
                            | from_base16(input[3]),
                    ],
                ))
            }
        } else {
            None
        }
    }
}

pub struct DecInt;

impl Combinator<i32> for DecInt {
    fn serialize(&self, input: &[i32]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 || input[0] < 0 {
            None
        } else {
            let mut ret = vec![];
            let mut v = input[0];
            if v == 0 {
                ret.push('0' as u8);
            } else {
                while v > 0 {
                    ret.push((v % 10) as u8 + '0' as u8);
                    v /= 10;
                }
                ret.reverse();
            }
            Some((1, ret))
        }
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<i32>)> {
        let mut size = 0;
        let mut ret = 0;
        while size < input.len() && is_dec(input[size]) {
            ret = ret * 10 + (input[size] - '0' as u8) as i32;
            size += 1;
        }
        if size == 0 {
            None
        } else {
            Some((size, vec![ret]))
        }
    }
}

pub struct Map<C, F, G> {
    base_serializer: C,
    a_to_b: F,
    b_to_a: G,
}

impl<C, F, G> Map<C, F, G> {
    pub fn new<A, B>(base_serializer: C, a_to_b: F, b_to_a: G) -> Map<C, F, G>
    where
        A: Clone,
        C: Combinator<B>,
        F: Fn(A) -> Option<B>,
        G: Fn(B) -> Option<A>,
    {
        Map {
            base_serializer,
            a_to_b,
            b_to_a,
        }
    }
}

impl<A, B, C, F, G> Combinator<A> for Map<C, F, G>
where
    A: Clone,
    C: Combinator<B>,
    F: Fn(A) -> Option<B>,
    G: Fn(B) -> Option<A>,
{
    fn serialize(&self, input: &[A]) -> Option<(usize, Vec<u8>)> {
        let mut input_conv = vec![];
        for a in input {
            if let Some(b) = (self.a_to_b)(a.clone()) {
                input_conv.push(b);
            } else {
                break;
            }
        }
        self.base_serializer.serialize(&input_conv)
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<A>)> {
        let (n_read, data) = self.base_serializer.deserialize(input)?;
        let data = data
            .into_iter()
            .map(|b| (self.b_to_a)(b))
            .collect::<Vec<_>>();
        if data.iter().any(|x| x.is_none()) {
            return None;
        }
        Some((
            n_read,
            data.into_iter().map(|x| x.unwrap()).collect::<Vec<_>>(),
        ))
    }
}

pub struct Optionalize<C>(C);

impl<C> Optionalize<C> {
    pub fn new(base_serializer: C) -> Optionalize<C> {
        Optionalize(base_serializer)
    }
}

impl<C, T> Combinator<Option<T>> for Optionalize<C>
where
    C: Combinator<T>,
    T: Clone,
{
    fn serialize(&self, input: &[Option<T>]) -> Option<(usize, Vec<u8>)> {
        Map::new(&self.0, |x| x, |x| Some(Some(x))).serialize(input)
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<Option<T>>)> {
        Map::new(&self.0, |x| x, |x| Some(Some(x))).deserialize(input)
    }
}

pub struct Seq<S> {
    base_serializer: S,
    count: usize,
}

impl<S> Seq<S> {
    pub fn new(base_serializer: S, count: usize) -> Seq<S> {
        Seq {
            base_serializer,
            count,
        }
    }
}

impl<S, T> Combinator<Vec<T>> for Seq<S>
where
    S: Combinator<T>,
{
    fn serialize(&self, input: &[Vec<T>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }

        let data = &input[0];

        let mut ofs = 0;
        let mut ret = vec![];
        for _ in 0..self.count {
            let (n_read, part) = self.base_serializer.serialize(&data[ofs..])?;
            ofs += n_read;
            ret.extend(part);
        }

        Some((1, ret))
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<Vec<T>>)> {
        let mut ofs = 0;
        let mut ret = vec![];
        while ret.len() < self.count {
            let (n_read, part) = self.base_serializer.deserialize(&input[ofs..])?;
            ofs += n_read;
            ret.extend(part);
        }

        ret.truncate(self.count);
        Some((ofs, vec![ret]))
    }
}

pub struct Grid<S> {
    base_serializer: S,
}

impl<S> Grid<S> {
    pub fn new(base_serializer: S) -> Grid<S> {
        Grid { base_serializer }
    }
}

impl<S, T> Combinator<Vec<Vec<T>>> for Grid<S>
where
    S: Combinator<T>,
    T: Clone,
{
    fn serialize(&self, input: &[Vec<Vec<T>>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }

        let data = &input[0];
        let height = data.len();
        assert!(height > 0);
        let width = data[0].len();
        for i in 1..height {
            assert_eq!(data[i].len(), width);
        }

        let mut ret = vec![];
        let (_, app) = DecInt.serialize(&[width as i32])?;
        ret.extend(app);
        ret.push('/' as u8);
        let (_, app) = DecInt.serialize(&[height as i32])?;
        ret.extend(app);
        ret.push('/' as u8);

        let mut input_flat = vec![];
        for row in data {
            input_flat.extend(row.clone());
        }

        let seq_combinator = Seq::new(&self.base_serializer, height * width);
        let (n_read, app) = seq_combinator.serialize(&[input_flat])?;
        if n_read == 1 {
            ret.extend(app);
            Some((1, ret))
        } else {
            None
        }
    }

    fn deserialize(&self, input: &[u8]) -> Option<(usize, Vec<Vec<Vec<T>>>)> {
        let mut n_read_total = 0;

        let (n_read, width) = DecInt.deserialize(input)?;
        assert_eq!(width.len(), 1);
        n_read_total += n_read;
        let width = width[0] as usize;
        if input.len() == n_read_total || input[n_read_total] != '/' as u8 {
            return None;
        }
        n_read_total += 1;

        let (n_read, height) = DecInt.deserialize(&input[n_read_total..])?;
        assert_eq!(height.len(), 1);
        n_read_total += n_read;
        let height = height[0] as usize;
        if input.len() == n_read_total || input[n_read_total] != '/' as u8 {
            return None;
        }
        n_read_total += 1;

        let seq_combinator = Seq::new(&self.base_serializer, height * width);
        let (n_read, ret_flat) = seq_combinator.deserialize(&input[n_read_total..])?;
        assert_eq!(ret_flat.len(), 1);
        let ret_flat = ret_flat.into_iter().next().unwrap();
        if ret_flat.len() != height * width {
            return None;
        }
        n_read_total += n_read;
        let mut ret = vec![];
        for i in 0..height {
            ret.push(
                ret_flat[(i * width)..((i + 1) * width)]
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        }
        Some((n_read_total, vec![ret]))
    }
}

pub fn problem_to_url<T, C>(combinator: C, puzzle_kind: &str, problem: T) -> Option<String>
where
    C: Combinator<T>,
{
    let (_, body) = combinator.serialize(&[problem])?;
    let prefix = String::from("https://puzz.link/p?");
    String::from_utf8(body)
        .ok()
        .map(|body| prefix + puzzle_kind + &body)
}

pub fn url_to_problem<T, C>(combinator: C, puzzle_kinds: &[&str], serialized: &str) -> Option<T>
where
    C: Combinator<T>,
{
    let serialized = serialized
        .strip_prefix("http://")
        .or(serialized.strip_prefix("https://"))?;
    let serialized = serialized
        .strip_prefix("puzz.link/p?")
        .or(serialized.strip_prefix("pzv.jp/p.html?"))?;
    let pos = serialized.find('/')?;
    let kind = &serialized[0..pos];
    if !puzzle_kinds.iter().any(|&k| kind == k) {
        return None;
    }
    let body = &serialized[(pos + 1)..];
    let (_, mut problem) = combinator.deserialize(body.as_bytes())?;
    assert_eq!(problem.len(), 1);
    problem.pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spaces() {
        let combinator = Spaces::new(0i32, 'w');

        assert_eq!(combinator.serialize(&[]), None);
        assert_eq!(
            combinator.serialize(&[0, 0, 1, 2]),
            Some((2, Vec::from("x")))
        );
        assert_eq!(combinator.serialize(&[1, 2, 3]), None);
        assert_eq!(
            combinator.serialize(&[0, 0, 0, 0, 0, 1]),
            Some((4, Vec::from("z")))
        );

        assert_eq!(
            combinator.deserialize("x".as_bytes()),
            Some((1, vec![0, 0]))
        );
        assert_eq!(combinator.deserialize("b".as_bytes()), None);
        assert_eq!(combinator.deserialize("".as_bytes()), None);
        assert_eq!(
            combinator.deserialize("zz".as_bytes()),
            Some((1, vec![0, 0, 0, 0]))
        );
    }

    #[test]
    fn test_hexint() {
        let combinator = HexInt;

        assert_eq!(combinator.serialize(&[]), None);
        assert_eq!(combinator.serialize(&[12, 3]), Some((1, Vec::from("c"))));
        assert_eq!(combinator.serialize(&[0, 5]), Some((1, Vec::from("0"))));
        assert_eq!(combinator.serialize(&[-1, 5]), None);
        assert_eq!(combinator.serialize(&[42]), Some((1, Vec::from("-2a"))));
        assert_eq!(combinator.serialize(&[1000]), Some((1, Vec::from("+3e8"))));
        assert_eq!(combinator.serialize(&[4096]), None);

        assert_eq!(combinator.deserialize("".as_bytes()), None);
        assert_eq!(combinator.deserialize("c".as_bytes()), Some((1, vec![12])));
        assert_eq!(combinator.deserialize("0".as_bytes()), Some((1, vec![0])));
        assert_eq!(
            combinator.deserialize("-2a11".as_bytes()),
            Some((3, vec![42]))
        );
        assert_eq!(
            combinator.deserialize("+3e85".as_bytes()),
            Some((4, vec![1000]))
        );
    }

    #[test]
    fn test_optionalize() {
        let combinator = Optionalize(HexInt);

        assert_eq!(combinator.serialize(&[]), None);
        assert_eq!(
            combinator.serialize(&[Some(12), Some(3)]),
            Some((1, Vec::from("c")))
        );
        assert_eq!(
            combinator.serialize(&[Some(0), None]),
            Some((1, Vec::from("0")))
        );
        assert_eq!(combinator.serialize(&[None, Some(5)]), None);

        assert_eq!(combinator.deserialize("".as_bytes()), None);
        assert_eq!(
            combinator.deserialize("c".as_bytes()),
            Some((1, vec![Some(12)]))
        );
        assert_eq!(
            combinator.deserialize("0".as_bytes()),
            Some((1, vec![Some(0)]))
        );
    }

    #[test]
    fn test_choice() {
        let combinator = Choice::new(vec![
            Box::new(Optionalize::new(HexInt)),
            Box::new(Spaces::new(None, 'g')),
        ]);

        assert_eq!(combinator.serialize(&[]), None);
        assert_eq!(
            combinator.serialize(&[Some(12), Some(3)]),
            Some((1, Vec::from("c")))
        );
        assert_eq!(
            combinator.serialize(&[Some(0), None]),
            Some((1, Vec::from("0")))
        );
        assert_eq!(
            combinator.serialize(&[None, None, Some(5)]),
            Some((2, Vec::from("h")))
        );

        assert_eq!(combinator.deserialize("".as_bytes()), None);
        assert_eq!(
            combinator.deserialize("c".as_bytes()),
            Some((1, vec![Some(12)]))
        );
        assert_eq!(
            combinator.deserialize("0".as_bytes()),
            Some((1, vec![Some(0)]))
        );
        assert_eq!(
            combinator.deserialize("h".as_bytes()),
            Some((1, vec![None, None]))
        );
    }

    #[test]
    fn test_seq() {
        let combinator = Seq::new(HexInt, 3);

        assert_eq!(combinator.serialize(&[vec![1, 2]]), None);
        assert_eq!(
            combinator.serialize(&[vec![1, 2, 42, 8]]),
            Some((1, Vec::from("12-2a")))
        );

        assert_eq!(combinator.deserialize("12".as_bytes()), None);
        assert_eq!(combinator.deserialize("12-2".as_bytes()), None);
        assert_eq!(
            combinator.deserialize("12-2a5".as_bytes()),
            Some((5, vec![vec![1, 2, 42]]))
        );
    }

    #[test]
    fn test_grid() {
        let combinator = Grid::new(HexInt);

        assert_eq!(
            combinator.serialize(&[vec![vec![2, 3, 1], vec![42, 1, 0],]]),
            Some((1, Vec::from("3/2/231-2a10")))
        );
        assert_eq!(
            combinator.serialize(&[vec![vec![2, 3], vec![42, -1],]]),
            None
        );

        assert_eq!(
            combinator.deserialize("3/2/231-2a10".as_bytes()),
            Some((12, vec![vec![vec![2, 3, 1], vec![42, 1, 0],],]))
        );
        assert_eq!(combinator.deserialize("3/3/231-2a10".as_bytes()), None);
    }
}
