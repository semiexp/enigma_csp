pub fn is_dec(c: u8) -> bool {
    return '0' as u8 <= c && c <= '9' as u8;
}

pub fn is_hex(c: u8) -> bool {
    return ('0' as u8 <= c && c <= '9' as u8) || ('a' as u8 <= c && c <= 'f' as u8);
}

pub fn is_base36(c: u8) -> bool {
    return ('0' as u8 <= c && c <= '9' as u8) || ('a' as u8 <= c && c <= 'z' as u8);
}

pub fn to_base36(n: i32) -> u8 {
    assert!(0 <= n && n < 36);
    if n <= 9 {
        n as u8 + '0' as u8
    } else {
        (n - 10) as u8 + 'a' as u8
    }
}

pub fn from_base36(c: u8) -> i32 {
    assert!(is_base36(c));
    if '0' as u8 <= c && c <= '9' as u8 {
        (c - '0' as u8) as i32
    } else {
        (c - 'a' as u8) as i32 + 10
    }
}

pub fn to_base16(n: i32) -> u8 {
    assert!(0 <= n && n < 16);
    to_base36(n)
}

pub fn from_base16(c: u8) -> i32 {
    assert!(is_hex(c));
    from_base36(c)
}

pub struct Context {
    pub height: Option<usize>,
    pub width: Option<usize>,
}

impl Context {
    pub fn new() -> Context {
        Context {
            height: None,
            width: None,
        }
    }
}

pub trait Combinator<T> {
    fn serialize(&self, ctx: &Context, input: &[T]) -> Option<(usize, Vec<u8>)>;
    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<T>)>;
}

impl<A, T> Combinator<T> for &A
where
    A: Combinator<T>,
{
    fn serialize(&self, ctx: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
        (*self).serialize(ctx, input)
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        (*self).deserialize(ctx, input)
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
    fn serialize(&self, ctx: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
        self.choices
            .iter()
            .find_map(|choice| choice.serialize(ctx, input))
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        self.choices
            .iter()
            .find_map(|choice| choice.deserialize(ctx, input))
    }
}

pub struct Dict<T> {
    before: T,
    after: Vec<u8>,
}

impl<T: Clone + PartialEq> Dict<T> {
    pub fn new<'a, A>(before: T, after: A) -> Dict<T>
    where
        Vec<u8>: From<A>,
    {
        Dict {
            before,
            after: Vec::<u8>::from(after),
        }
    }
}

impl<T: Clone + PartialEq> Combinator<T> for Dict<T> {
    fn serialize(&self, _: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
        if input.len() > 0 && input[0] == self.before {
            Some((1, self.after.clone()))
        } else {
            None
        }
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        if input.len() >= self.after.len() && input[..self.after.len()] == self.after {
            Some((self.after.len(), vec![self.before.clone()]))
        } else {
            None
        }
    }
}

pub struct MaybeSkip<C>(Vec<u8>, C);

impl<C> MaybeSkip<C> {
    pub fn new<A>(s: A, combinator: C) -> MaybeSkip<C>
    where
        Vec<u8>: From<A>,
    {
        MaybeSkip(Vec::<u8>::from(s), combinator)
    }
}

impl<T, C> Combinator<T> for MaybeSkip<C>
where
    C: Combinator<T>,
{
    fn serialize(&self, ctx: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
        self.1.serialize(ctx, input)
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        if input.len() >= self.0.len() && input[..self.0.len()] == self.0 {
            let (n_read, res) = self.1.deserialize(ctx, &input[self.0.len()..])?;
            Some((n_read + self.0.len(), res))
        } else {
            self.1.deserialize(ctx, input)
        }
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
    fn serialize(&self, _: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
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

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
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
    fn serialize(&self, _: &Context, input: &[i32]) -> Option<(usize, Vec<u8>)> {
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

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<i32>)> {
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
    fn serialize(&self, _: &Context, input: &[i32]) -> Option<(usize, Vec<u8>)> {
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

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<i32>)> {
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

pub struct MultiDigit {
    base: i32,
    num_digits: usize,
    max_num: i32,
}

impl MultiDigit {
    pub fn new(base: i32, num_digits: usize) -> MultiDigit {
        assert!(2 <= base);
        assert!(1 <= num_digits);
        let max_num = base.pow(num_digits as u32);
        assert!(max_num <= 36);
        MultiDigit {
            base,
            num_digits,
            max_num,
        }
    }
}

impl Combinator<i32> for MultiDigit {
    fn serialize(&self, _: &Context, input: &[i32]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let mut v = 0;
        let mut n_read = 0;
        for i in 0..self.num_digits {
            if i >= input.len() || !(0 <= input[i] && input[i] < self.base) {
                break;
            }
            n_read += 1;
            v += input[i] * self.base.pow((self.num_digits - 1 - i) as u32);
        }
        if n_read == 0 {
            None
        } else {
            Some((n_read, vec![to_base36(v)]))
        }
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<i32>)> {
        if input.len() == 0 {
            return None;
        }
        let v = input[0];
        if !is_base36(v) {
            return None;
        }
        let mut v = from_base36(v);
        if v >= self.max_num {
            return None;
        }
        let mut ret = vec![];
        for _ in 0..self.num_digits {
            ret.push(v % self.base);
            v /= self.base;
        }
        ret.reverse();
        Some((1, ret))
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
    fn serialize(&self, ctx: &Context, input: &[A]) -> Option<(usize, Vec<u8>)> {
        let mut input_conv = vec![];
        for a in input {
            if let Some(b) = (self.a_to_b)(a.clone()) {
                input_conv.push(b);
            } else {
                break;
            }
        }
        self.base_serializer.serialize(ctx, &input_conv)
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<A>)> {
        let (n_read, data) = self.base_serializer.deserialize(ctx, input)?;
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
    fn serialize(&self, ctx: &Context, input: &[Option<T>]) -> Option<(usize, Vec<u8>)> {
        Map::new(&self.0, |x| x, |x| Some(Some(x))).serialize(ctx, input)
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Option<T>>)> {
        Map::new(&self.0, |x| x, |x| Some(Some(x))).deserialize(ctx, input)
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
    fn serialize(&self, ctx: &Context, input: &[Vec<T>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }

        let data = &input[0];

        let mut ofs = 0;
        let mut ret = vec![];
        while ofs < self.count {
            let (n_read, part) = self.base_serializer.serialize(ctx, &data[ofs..])?;
            ofs += n_read;
            ret.extend(part);
        }

        Some((1, ret))
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Vec<T>>)> {
        let mut ofs = 0;
        let mut ret = vec![];
        while ret.len() < self.count {
            let (n_read, part) = self.base_serializer.deserialize(ctx, &input[ofs..])?;
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
    fn serialize(&self, ctx: &Context, input: &[Vec<Vec<T>>]) -> Option<(usize, Vec<u8>)> {
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
        let (_, app) = DecInt.serialize(ctx, &[width as i32])?;
        ret.extend(app);
        ret.push('/' as u8);
        let (_, app) = DecInt.serialize(ctx, &[height as i32])?;
        ret.extend(app);
        ret.push('/' as u8);

        let mut input_flat = vec![];
        for row in data {
            input_flat.extend(row.clone());
        }

        let seq_combinator = Seq::new(&self.base_serializer, height * width);
        let (n_read, app) = seq_combinator.serialize(ctx, &[input_flat])?;
        if n_read == 1 {
            ret.extend(app);
            Some((1, ret))
        } else {
            None
        }
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Vec<Vec<T>>>)> {
        let mut n_read_total = 0;

        let (n_read, width) = DecInt.deserialize(ctx, input)?;
        assert_eq!(width.len(), 1);
        n_read_total += n_read;
        let width = width[0] as usize;
        if input.len() == n_read_total || input[n_read_total] != '/' as u8 {
            return None;
        }
        n_read_total += 1;

        let (n_read, height) = DecInt.deserialize(ctx, &input[n_read_total..])?;
        assert_eq!(height.len(), 1);
        n_read_total += n_read;
        let height = height[0] as usize;
        if input.len() == n_read_total || input[n_read_total] != '/' as u8 {
            return None;
        }
        n_read_total += 1;

        let seq_combinator = Seq::new(&self.base_serializer, height * width);
        let (n_read, ret_flat) = seq_combinator.deserialize(ctx, &input[n_read_total..])?;
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
    let (_, body) = combinator.serialize(&Context::new(), &[problem])?;
    let prefix = String::from("https://puzz.link/p?");
    String::from_utf8(body)
        .ok()
        .map(|body| prefix + puzzle_kind + "/" + &body)
}

pub fn url_to_puzzle_kind(serialized: &str) -> Option<String> {
    let serialized = serialized
        .strip_prefix("http://")
        .or(serialized.strip_prefix("https://"))?;
    let serialized = serialized
        .strip_prefix("puzz.link/p?")
        .or(serialized.strip_prefix("pzv.jp/p.html?"))?;
    let pos = serialized.find('/')?;
    let kind = &serialized[0..pos];
    Some(String::from(kind))
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
    let (_, mut problem) = combinator.deserialize(&Context::new(), body.as_bytes())?;
    assert_eq!(problem.len(), 1);
    problem.pop()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict() {
        let ctx = &Context::new();
        let combinator = Dict::new(42, "ab");

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[42, 0]),
            Some((1, Vec::from("ab")))
        );

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(combinator.deserialize(ctx, "aaa".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "aba".as_bytes()),
            Some((2, vec![42]))
        );
    }

    #[test]
    fn test_spaces() {
        let ctx = &Context::new();
        let combinator = Spaces::new(0i32, 'w');

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[0, 0, 1, 2]),
            Some((2, Vec::from("x")))
        );
        assert_eq!(combinator.serialize(ctx, &[1, 2, 3]), None);
        assert_eq!(
            combinator.serialize(ctx, &[0, 0, 0, 0, 0, 1]),
            Some((4, Vec::from("z")))
        );

        assert_eq!(
            combinator.deserialize(ctx, "x".as_bytes()),
            Some((1, vec![0, 0]))
        );
        assert_eq!(combinator.deserialize(ctx, "b".as_bytes()), None);
        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "zz".as_bytes()),
            Some((1, vec![0, 0, 0, 0]))
        );
    }

    #[test]
    fn test_hexint() {
        let ctx = &Context::new();
        let combinator = HexInt;

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[12, 3]),
            Some((1, Vec::from("c")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[0, 5]),
            Some((1, Vec::from("0")))
        );
        assert_eq!(combinator.serialize(ctx, &[-1, 5]), None);
        assert_eq!(
            combinator.serialize(ctx, &[42]),
            Some((1, Vec::from("-2a")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[1000]),
            Some((1, Vec::from("+3e8")))
        );
        assert_eq!(combinator.serialize(ctx, &[4096]), None);

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "c".as_bytes()),
            Some((1, vec![12]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "0".as_bytes()),
            Some((1, vec![0]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "-2a11".as_bytes()),
            Some((3, vec![42]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "+3e85".as_bytes()),
            Some((4, vec![1000]))
        );
    }

    #[test]
    fn test_maybe_skip() {
        let ctx = &Context::new();
        let combinator = MaybeSkip::new("!!", HexInt);

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[42]),
            Some((1, Vec::from("-2a")))
        );
        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "c".as_bytes()),
            Some((1, vec![12]))
        );
        assert_eq!(combinator.deserialize(ctx, "!!".as_bytes()), None);
        assert_eq!(combinator.deserialize(ctx, "!c".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "!!c".as_bytes()),
            Some((3, vec![12]))
        );
    }

    #[test]
    fn test_multi_digit() {
        let ctx = &Context::new();
        let combinator = MultiDigit::new(3, 3usize);

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[1, 0, 2]),
            Some((3, Vec::from("b")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[2, 0]),
            Some((2, Vec::from("i")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[1, 0, 3]),
            Some((2, Vec::from("9")))
        );
        assert_eq!(combinator.serialize(ctx, &[3, 1, 0]), None);

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "0".as_bytes()),
            Some((1, vec![0, 0, 0]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "b".as_bytes()),
            Some((1, vec![1, 0, 2]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "q".as_bytes()),
            Some((1, vec![2, 2, 2]))
        );
        assert_eq!(combinator.deserialize(ctx, "r".as_bytes()), None);
    }

    #[test]
    fn test_optionalize() {
        let ctx = &Context::new();
        let combinator = Optionalize(HexInt);

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[Some(12), Some(3)]),
            Some((1, Vec::from("c")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[Some(0), None]),
            Some((1, Vec::from("0")))
        );
        assert_eq!(combinator.serialize(ctx, &[None, Some(5)]), None);

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "c".as_bytes()),
            Some((1, vec![Some(12)]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "0".as_bytes()),
            Some((1, vec![Some(0)]))
        );
    }

    #[test]
    fn test_choice() {
        let ctx = &Context::new();
        let combinator = Choice::new(vec![
            Box::new(Optionalize::new(HexInt)),
            Box::new(Spaces::new(None, 'g')),
        ]);

        assert_eq!(combinator.serialize(ctx, &[]), None);
        assert_eq!(
            combinator.serialize(ctx, &[Some(12), Some(3)]),
            Some((1, Vec::from("c")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[Some(0), None]),
            Some((1, Vec::from("0")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[None, None, Some(5)]),
            Some((2, Vec::from("h")))
        );

        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "c".as_bytes()),
            Some((1, vec![Some(12)]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "0".as_bytes()),
            Some((1, vec![Some(0)]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "h".as_bytes()),
            Some((1, vec![None, None]))
        );
    }

    #[test]
    fn test_seq() {
        let ctx = &Context::new();
        let combinator = Seq::new(HexInt, 3);

        assert_eq!(combinator.serialize(ctx, &[vec![1, 2]]), None);
        assert_eq!(
            combinator.serialize(ctx, &[vec![1, 2, 42, 8]]),
            Some((1, Vec::from("12-2a")))
        );

        assert_eq!(combinator.deserialize(ctx, "12".as_bytes()), None);
        assert_eq!(combinator.deserialize(ctx, "12-2".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "12-2a5".as_bytes()),
            Some((5, vec![vec![1, 2, 42]]))
        );
    }

    #[test]
    fn test_grid() {
        let ctx = &Context::new();
        let combinator = Grid::new(HexInt);

        assert_eq!(
            combinator.serialize(ctx, &[vec![vec![2, 3, 1], vec![42, 1, 0],]]),
            Some((1, Vec::from("3/2/231-2a10")))
        );
        assert_eq!(
            combinator.serialize(ctx, &[vec![vec![2, 3], vec![42, -1],]]),
            None
        );

        assert_eq!(
            combinator.deserialize(ctx, "3/2/231-2a10".as_bytes()),
            Some((12, vec![vec![vec![2, 3, 1], vec![42, 1, 0],],]))
        );
        assert_eq!(combinator.deserialize(ctx, "3/3/231-2a10".as_bytes()), None);
    }
}
