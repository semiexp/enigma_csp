use crate::graph::{borders_to_rooms, GridFrame};

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

    pub fn sized(height: usize, width: usize) -> Context {
        Context {
            height: Some(height),
            width: Some(width),
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

pub struct Sequencer<'a, T> {
    input: &'a [T],
    n_read: usize,
}

impl<'a, T> Sequencer<'a, T> {
    pub fn new(input: &'a [T]) -> Sequencer<'a, T> {
        Sequencer { input, n_read: 0 }
    }

    pub fn n_read(&self) -> usize {
        self.n_read
    }

    pub fn serialize<C: Combinator<T>>(&mut self, ctx: &Context, combinator: C) -> Option<Vec<u8>> {
        if let Some((n, res)) = combinator.serialize(ctx, &self.input[self.n_read..]) {
            self.n_read += n;
            Some(res)
        } else {
            None
        }
    }
}

impl<'a> Sequencer<'a, u8> {
    pub fn deserialize<T, C: Combinator<T>>(
        &mut self,
        ctx: &Context,
        combinator: C,
    ) -> Option<Vec<T>> {
        if let Some((n, res)) = combinator.deserialize(ctx, &self.input[self.n_read..]) {
            self.n_read += n;
            Some(res)
        } else {
            None
        }
    }

    pub fn deserialize_one_elem<T, C: Combinator<T>>(
        &mut self,
        ctx: &Context,
        combinator: C,
    ) -> Option<T> {
        let t = self.deserialize(ctx, combinator)?;
        assert_eq!(t.len(), 1);
        t.into_iter().next()
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
    minimum: i32,
    maximum: i32,
}

impl<T: Clone + PartialEq> Spaces<T> {
    pub fn new(space: T, minimum: char) -> Spaces<T> {
        Spaces {
            space,
            minimum: from_base36(minimum as u8),
            maximum: from_base36('z' as u8),
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
            Some((
                n_spaces,
                vec![to_base36(self.minimum + (n_spaces as i32 - 1))],
            ))
        }
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        if input.len() == 0 {
            return None;
        }
        let v = input[0];
        if !is_base36(v) {
            return None;
        }
        let v = from_base36(v);
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

pub struct FixedLengthHexInt(usize);

impl FixedLengthHexInt {
    pub fn new(len: usize) -> FixedLengthHexInt {
        assert!(1 <= len && len <= 7);
        FixedLengthHexInt(len)
    }
}

impl Combinator<i32> for FixedLengthHexInt {
    fn serialize(&self, _: &Context, input: &[i32]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let v = input[0];
        let len = self.0;
        let lo = if len == 1 { 0 } else { 1i32 << ((len - 1) * 4) };
        let hi = (1i32 << (len * 4)) - 1;
        if lo <= v && v <= hi {
            let mut ret = vec![];
            for i in 0..len {
                ret.push(to_base16((v >> (4 * (len - 1 - i))) & 15));
            }
            Some((1, ret))
        } else {
            None
        }
    }

    fn deserialize(&self, _: &Context, input: &[u8]) -> Option<(usize, Vec<i32>)> {
        let len = self.0;
        if input.len() < len {
            return None;
        }
        let mut ret = 0;
        for i in 0..len {
            if !is_hex(input[i]) {
                return None;
            }
            ret = (ret << 4) | from_base16(input[i]);
        }
        let lo = if len == 1 { 0 } else { 1i32 << ((len - 1) * 4) };
        let hi = (1i32 << (len * 4)) - 1;
        if lo <= ret && ret <= hi {
            Some((len, vec![ret]))
        } else {
            None
        }
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

        let mut sequencer = Sequencer::new(&input[0]);
        let mut ret = vec![];
        while sequencer.n_read() < self.count {
            let part = sequencer.serialize(ctx, &self.base_serializer)?;
            ret.extend(part);
        }

        Some((1, ret))
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Vec<T>>)> {
        let mut sequencer = Sequencer::new(input);
        let mut ret = vec![];
        while ret.len() < self.count {
            let part = sequencer.deserialize(ctx, &self.base_serializer)?;
            ret.extend(part);
        }

        ret.truncate(self.count);
        Some((sequencer.n_read(), vec![ret]))
    }
}

pub struct ContextBasedGrid<S> {
    base_serializer: S,
}

impl<S> ContextBasedGrid<S> {
    pub fn new(base_serializer: S) -> ContextBasedGrid<S> {
        ContextBasedGrid { base_serializer }
    }
}

impl<S, T> Combinator<Vec<Vec<T>>> for ContextBasedGrid<S>
where
    S: Combinator<T>,
    T: Clone,
{
    fn serialize(&self, ctx: &Context, input: &[Vec<Vec<T>>]) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }

        let data = &input[0];
        let height = ctx.height.unwrap();
        assert_eq!(data.len(), height);
        let width = ctx.width.unwrap();
        for i in 0..height {
            assert_eq!(data[i].len(), width);
        }

        let mut input_flat = vec![];
        for row in data {
            input_flat.extend(row.clone());
        }

        let seq_combinator = Seq::new(&self.base_serializer, height * width);
        let (n_read, ret) = seq_combinator.serialize(ctx, &[input_flat])?;
        if n_read == 1 {
            Some((1, ret))
        } else {
            None
        }
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Vec<Vec<T>>>)> {
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let seq_combinator = Seq::new(&self.base_serializer, height * width);
        let (n_read, ret_flat) = seq_combinator.deserialize(ctx, input)?;
        assert_eq!(ret_flat.len(), 1);
        let ret_flat = ret_flat.into_iter().next().unwrap();
        if ret_flat.len() != height * width {
            return None;
        }
        let mut ret = vec![];
        for i in 0..height {
            ret.push(
                ret_flat[(i * width)..((i + 1) * width)]
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        }
        Some((n_read, vec![ret]))
    }
}

pub struct Size<S> {
    base_serializer: S,
}

impl<S> Size<S> {
    pub fn new(base_serializer: S) -> Size<S> {
        Size { base_serializer }
    }
}

impl<S, T> Combinator<T> for Size<S>
where
    S: Combinator<T>,
{
    fn serialize(&self, ctx: &Context, input: &[T]) -> Option<(usize, Vec<u8>)> {
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let mut ret = vec![];
        let (_, app) = DecInt.serialize(ctx, &[width as i32])?;
        ret.extend(app);
        ret.push('/' as u8);
        let (_, app) = DecInt.serialize(ctx, &[height as i32])?;
        ret.extend(app);
        ret.push('/' as u8);

        let (n_read, app) = self.base_serializer.serialize(ctx, input)?;
        ret.extend(app);

        Some((n_read, ret))
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<T>)> {
        let mut sequencer = Sequencer::new(input);

        let width = sequencer.deserialize(ctx, DecInt)?;
        assert_eq!(width.len(), 1);
        let width = width[0] as usize;
        sequencer.deserialize(ctx, Dict::new((), "/"))?;

        let height = sequencer.deserialize(ctx, DecInt)?;
        assert_eq!(height.len(), 1);
        let height = height[0] as usize;
        sequencer.deserialize(ctx, Dict::new((), "/"))?;

        let ctx = Context {
            height: Some(height),
            width: Some(width),
            ..*ctx
        };
        let ret = sequencer.deserialize(&ctx, &self.base_serializer)?;
        Some((sequencer.n_read(), ret))
    }
}

fn map_2d<'a, A, B, F>(input: &'a Vec<Vec<A>>, func: F) -> Vec<Vec<B>>
where
    F: Fn(&'a A) -> B,
{
    input
        .iter()
        .map(|row| row.iter().map(|x| func(x)).collect())
        .collect()
}

pub struct Rooms;

impl Combinator<GridFrame<Vec<Vec<bool>>>> for Rooms {
    fn serialize(
        &self,
        ctx: &Context,
        input: &[GridFrame<Vec<Vec<bool>>>],
    ) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();

        let vertical_i32 = map_2d(&input[0].vertical, |&b| if b { 1 } else { 0 });
        let horizontal_i32 = map_2d(&input[0].horizontal, |&b| if b { 1 } else { 0 });

        let ctx_grid = ContextBasedGrid::new(MultiDigit::new(2, 5));
        let mut ret = vec![];
        let (_, app) = ctx_grid.serialize(
            &Context {
                height: Some(height),
                width: Some(width - 1),
                ..*ctx
            },
            &vec![vertical_i32],
        )?;
        ret.extend(app);
        let (_, app) = ctx_grid.serialize(
            &Context {
                height: Some(height - 1),
                width: Some(width),
                ..*ctx
            },
            &vec![horizontal_i32],
        )?;
        ret.extend(app);

        Some((1, ret))
    }

    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<GridFrame<Vec<Vec<bool>>>>)> {
        let height = ctx.height.unwrap();
        let width = ctx.width.unwrap();
        let mut sequencer = Sequencer::new(input);
        let ctx_grid = ContextBasedGrid::new(MultiDigit::new(2, 5));

        let vertical_i32 = sequencer.deserialize_one_elem(
            &Context {
                height: Some(height),
                width: Some(width - 1),
                ..*ctx
            },
            &ctx_grid,
        )?;
        let horizontal_i32 = sequencer.deserialize_one_elem(
            &Context {
                height: Some(height - 1),
                width: Some(width),
                ..*ctx
            },
            &ctx_grid,
        )?;

        let vertical = map_2d(&vertical_i32, |&n| n == 1);
        let horizontal = map_2d(&horizontal_i32, |&n| n == 1);

        Some((
            sequencer.n_read(),
            vec![GridFrame {
                vertical,
                horizontal,
            }],
        ))
    }
}

pub struct RoomsWithValues<C> {
    value_combinator: C,
}

impl<C> RoomsWithValues<C> {
    pub fn new(value_combinator: C) -> RoomsWithValues<C> {
        RoomsWithValues { value_combinator }
    }
}

impl<T, C> Combinator<(GridFrame<Vec<Vec<bool>>>, Vec<T>)> for RoomsWithValues<C>
where
    T: Clone,
    C: Combinator<T>,
{
    fn serialize(
        &self,
        ctx: &Context,
        input: &[(GridFrame<Vec<Vec<bool>>>, Vec<T>)],
    ) -> Option<(usize, Vec<u8>)> {
        if input.len() == 0 {
            return None;
        }
        let (borders, seq) = &input[0];

        let mut ret = vec![];
        let (_, app) = Rooms.serialize(ctx, &[borders.clone()])?;
        ret.extend(app);
        let n_rooms = borders_to_rooms(borders).len();
        let (_, app) = Seq::new(&self.value_combinator, n_rooms).serialize(ctx, &[seq.clone()])?;
        ret.extend(app);

        Some((1, ret))
    }

    fn deserialize(
        &self,
        ctx: &Context,
        input: &[u8],
    ) -> Option<(usize, Vec<(GridFrame<Vec<Vec<bool>>>, Vec<T>)>)> {
        let mut sequencer = Sequencer::new(input);

        let borders = sequencer.deserialize_one_elem(ctx, Rooms)?;
        let n_rooms = borders_to_rooms(&borders).len();
        let seq = sequencer.deserialize_one_elem(ctx, Seq::new(&self.value_combinator, n_rooms))?;

        Some((sequencer.n_read(), vec![(borders, seq)]))
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

        let height = input[0].len();
        assert!(height > 0);
        let width = input[0][0].len();

        let ctx = Context {
            height: Some(height),
            width: Some(width),
            ..*ctx
        };

        Size::new(ContextBasedGrid::new(&self.base_serializer)).serialize(&ctx, input)
    }

    fn deserialize(&self, ctx: &Context, input: &[u8]) -> Option<(usize, Vec<Vec<Vec<T>>>)> {
        Size::new(ContextBasedGrid::new(&self.base_serializer)).deserialize(&ctx, input)
    }
}

pub fn problem_to_url_with_context<T, C>(
    combinator: C,
    puzzle_kind: &str,
    problem: T,
    ctx: &Context,
) -> Option<String>
where
    C: Combinator<T>,
{
    let (_, body) = combinator.serialize(ctx, &[problem])?;
    let prefix = String::from("https://puzz.link/p?");
    String::from_utf8(body)
        .ok()
        .map(|body| prefix + puzzle_kind + "/" + &body)
}

pub fn problem_to_url<T, C>(combinator: C, puzzle_kind: &str, problem: T) -> Option<String>
where
    C: Combinator<T>,
{
    problem_to_url_with_context(combinator, puzzle_kind, problem, &Context::new())
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

        let combinator = Spaces::new(0i32, '9');
        assert_eq!(
            combinator.serialize(ctx, &[0, 0]),
            Some((2, Vec::from("a")))
        );
        assert_eq!(
            combinator.deserialize(ctx, "b".as_bytes()),
            Some((1, vec![0, 0, 0]))
        );
    }

    #[test]
    fn test_fixed_hexint() {
        let ctx = &Context::new();

        let combinator = FixedLengthHexInt::new(1);

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
        assert_eq!(combinator.serialize(ctx, &[42]), None);
        assert_eq!(combinator.deserialize(ctx, "".as_bytes()), None);
        assert_eq!(
            combinator.deserialize(ctx, "c".as_bytes()),
            Some((1, vec![12]))
        );
        assert_eq!(
            combinator.deserialize(ctx, "0".as_bytes()),
            Some((1, vec![0]))
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

    #[test]
    fn test_rooms_with_values() {
        let ctx = &Context::sized(3, 4);
        let combinator = RoomsWithValues::new(HexInt);

        assert_eq!(
            combinator.serialize(
                ctx,
                &[(
                    GridFrame {
                        vertical: vec![
                            vec![true, false, true],
                            vec![true, false, true],
                            vec![false, true, false],
                        ],
                        horizontal: vec![
                            vec![false, true, true, false],
                            vec![true, false, true, false],
                        ],
                    },
                    vec![1, 0, 1, 2]
                )]
            ),
            Some((1, Vec::from("mkd81012")))
        );

        assert_eq!(
            combinator.deserialize(ctx, "mkd81012".as_bytes()),
            Some((
                8,
                vec![(
                    GridFrame {
                        vertical: vec![
                            vec![true, false, true],
                            vec![true, false, true],
                            vec![false, true, false],
                        ],
                        horizontal: vec![
                            vec![false, true, true, false],
                            vec![true, false, true, false],
                        ],
                    },
                    vec![1, 0, 1, 2]
                )]
            ))
        );
    }
}
