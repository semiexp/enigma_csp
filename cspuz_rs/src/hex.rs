// A module for hexagonal grid.
//
// ## Visual representation
//
// A hexagonal grid looks like this:
//   \|-- b --|/
//  a * * * * * d
// \ * * * * * *
//  * * * * * * * /
// / * * * * * * *
//    * * * * * *
//   c * * * * *
//      * * * *
//       * * *
//      /
//
// Here, a, b, c and d represent the dimensions of the grid.
// We note that
// - The length of the bottom edge is `b + d - c`, and
// - The length of the bottom-right edge is `a + c - d`.
//
// ## Coordinate system
// We identify a hexonal grid with an ordinary 2D grid (with missing cells) as follows:
// - The top and bottom edges of the hexagonal grid are mapped to the top and bottom edges of the 2D grid.
// - The top-left and bottom-right edges of the hexagonal grid are mapped to the left and right edges of the 2D grid.
// - Thus, the top-right and bottom-left edges of the hexagonal grid are parallel to a diagonal of a 2D-grid cell.
// The height of the 2D grid is `a + c - 1`, and the width of the 2D grid is `b + d - 1`.
//
// We use the following coordinate system:
// - The leftmost cell in the top row is (0, 0).
// - The right cell of (y, x) is (y, x - 1). Thus, the left cell of (y, x) is (y, x - 1).
// - The bottom-left cell of (y, x) is (y + 1, x). Thus, the top-right cell of (y, x) is (y - 1, x).
// - The bottom-right cell of (y, x) is (y + 1, x + 1). Thus, the top-left cell of (y, x) is (y - 1, x - 1).
//
//      (y - 1, x - 1)  (y - 1, x)
// (y, x - 1)      (y, x)        (y, x + 1)
//        (y + 1, x)    (y + 1, x + 1)

use std::cmp::{Eq, PartialEq};
use std::ops::{Index, IndexMut};

use crate::graph::Graph;
use crate::solver::{BoolVar, FromModel, FromOwnedPartialModel, Model, Solver};

const HEX_NEIGHBORS: [(i32, i32); 6] = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)];

#[derive(Clone, Debug)]
struct HexCellMapping {
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    cell_to_idx: Vec<Vec<Option<usize>>>,
    idx_to_cell: Vec<(usize, usize)>,
}

fn num_cells(dims: (usize, usize, usize, usize)) -> usize {
    let (a, b, c, d) = dims;
    let height = a + c - 1;
    let width = b + d - 1;

    height * width - c * (c - 1) / 2 - d * (d - 1) / 2
}

impl HexCellMapping {
    fn new(dims: (usize, usize, usize, usize)) -> HexCellMapping {
        let (a, b, c, d) = dims;
        let height = a + c - 1;
        let width = b + d - 1;

        assert!(b + d > c);
        assert!(a + c > d);

        let mut cell_to_idx = vec![vec![None; width]; height];
        let mut idx_to_cell = vec![];
        let mut idx = 0;

        for y in 0..height {
            for x in 0..width {
                let yi = y as i32;
                let xi = x as i32;

                let d = yi - xi;
                if -(b as i32) < d && d < a as i32 {
                    cell_to_idx[y][x] = Some(idx);
                    idx_to_cell.push((y, x));
                    idx += 1;
                }
            }
        }

        assert_eq!(idx, num_cells(dims));

        HexCellMapping {
            a,
            b,
            c,
            d,
            cell_to_idx,
            idx_to_cell,
        }
    }

    fn dims(&self) -> (usize, usize, usize, usize) {
        (self.a, self.b, self.c, self.d)
    }

    fn repr_dims(&self) -> (usize, usize) {
        (self.a + self.c - 1, self.b + self.d - 1)
    }

    fn is_valid_coord(&self, coord: (usize, usize)) -> bool {
        let (y, x) = coord;

        y < self.cell_to_idx.len()
            && x < self.cell_to_idx[0].len()
            && self.cell_to_idx[y][x].is_some()
    }

    fn is_valid_coord_offset(&self, coord: (usize, usize), offset: (i32, i32)) -> bool {
        let (y, x) = coord;
        let (dy, dx) = offset;

        let (ny, nx) = (y as i32 + dy, x as i32 + dx);

        if ny < 0 || nx < 0 {
            return false;
        }

        self.is_valid_coord((ny as usize, nx as usize))
    }
}

#[derive(Clone, Debug)]
pub struct HexGrid<T> {
    cell_mapping: HexCellMapping,
    data: Vec<T>,
}

impl<T: PartialEq> PartialEq for HexGrid<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq> Eq for HexGrid<T> {}

impl<T> HexGrid<T> {
    fn from_raw(dims: (usize, usize, usize, usize), data: Vec<T>) -> HexGrid<T>
    where
        T: Clone,
    {
        HexGrid {
            cell_mapping: HexCellMapping::new(dims),
            data,
        }
    }

    pub fn filled(dims: (usize, usize, usize, usize), fill: T) -> HexGrid<T>
    where
        T: Clone,
    {
        let num_cells = num_cells(dims);
        let data = vec![fill; num_cells];
        HexGrid::from_raw(dims, data)
    }

    pub fn from_grid(dims: (usize, usize, usize, usize), data: Vec<Vec<T>>) -> HexGrid<T>
    where
        T: Clone,
    {
        let mapping = HexCellMapping::new(dims);
        let mut data_flat = vec![];

        for &(y, x) in &mapping.idx_to_cell {
            data_flat.push(data[y][x].clone());
        }

        HexGrid {
            cell_mapping: mapping,
            data: data_flat,
        }
    }

    pub fn flatten(&self) -> &[T] {
        &self.data
    }

    pub fn dims(&self) -> (usize, usize, usize, usize) {
        self.cell_mapping.dims()
    }

    pub fn repr_dims(&self) -> (usize, usize) {
        self.cell_mapping.repr_dims()
    }

    pub fn is_valid_coord(&self, coord: (usize, usize)) -> bool {
        self.cell_mapping.is_valid_coord(coord)
    }

    pub fn is_valid_coord_offset(&self, coord: (usize, usize), offset: (i32, i32)) -> bool {
        self.cell_mapping.is_valid_coord_offset(coord, offset)
    }

    pub fn cells(&self) -> &[(usize, usize)] {
        &self.cell_mapping.idx_to_cell
    }

    pub fn get_or_offset(&self, coord: (usize, usize), offset: (i32, i32), default: T) -> T
    where
        T: Clone,
    {
        let (y, x) = coord;
        let (dy, dx) = offset;
        let (ny, nx) = ((y as i32 + dy) as usize, (x as i32 + dx) as usize);
        if self.is_valid_coord((ny, nx)) {
            self[(ny, nx)].clone()
        } else {
            default
        }
    }
}

impl<T> Index<(usize, usize)> for HexGrid<T> {
    type Output = T;

    fn index(&self, coord: (usize, usize)) -> &T {
        let (y, x) = coord;
        let idx = self.cell_mapping.cell_to_idx[y][x].unwrap();
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for HexGrid<T> {
    fn index_mut(&mut self, coord: (usize, usize)) -> &mut T {
        let (y, x) = coord;
        let idx = self.cell_mapping.cell_to_idx[y][x].unwrap();
        &mut self.data[idx]
    }
}

pub type BoolHexGrid = HexGrid<BoolVar>;
pub type BoolHexGridModel = HexGrid<bool>;
pub type BoolHexGridIrrefutableFacts = HexGrid<Option<bool>>;

impl BoolHexGrid {
    pub fn new(solver: &mut Solver, dims: (usize, usize, usize, usize)) -> BoolHexGrid {
        let num_cells = num_cells(dims);
        let data = (0..num_cells).map(|_| solver.bool_var()).collect();
        HexGrid::from_raw(dims, data)
    }

    pub fn representation(&self) -> (Vec<BoolVar>, Graph) {
        let mut graph = Graph::new(self.data.len());

        for &(y, x) in &self.cell_mapping.idx_to_cell {
            if self.is_valid_coord_offset((y, x), (0, 1)) {
                graph.add_edge(
                    self.cell_mapping.cell_to_idx[y][x].unwrap(),
                    self.cell_mapping.cell_to_idx[y][x + 1].unwrap(),
                );
            }
            if self.is_valid_coord_offset((y, x), (1, 0)) {
                graph.add_edge(
                    self.cell_mapping.cell_to_idx[y][x].unwrap(),
                    self.cell_mapping.cell_to_idx[y + 1][x].unwrap(),
                );
            }
            if self.is_valid_coord_offset((y, x), (1, 1)) {
                graph.add_edge(
                    self.cell_mapping.cell_to_idx[y][x].unwrap(),
                    self.cell_mapping.cell_to_idx[y + 1][x + 1].unwrap(),
                );
            }
        }

        (self.data.clone(), graph)
    }
}

impl FromModel for BoolHexGrid {
    type Output = BoolHexGridModel;

    fn from_model(&self, model: &Model) -> Self::Output {
        let data = self.data.iter().map(|v| model.get(v)).collect::<Vec<_>>();
        HexGrid::from_raw(self.dims(), data)
    }
}

impl FromOwnedPartialModel for BoolHexGrid {
    type Output = BoolHexGridIrrefutableFacts;
    type OutputUnwrap = BoolHexGridModel;

    fn from_irrefutable_facts(
        &self,
        irrefutable_facts: &crate::solver::OwnedPartialModel,
    ) -> Self::Output {
        let data = self
            .data
            .iter()
            .map(|v| irrefutable_facts.get(v))
            .collect::<Vec<_>>();
        HexGrid::from_raw(self.dims(), data)
    }

    fn from_irrefutable_facts_unwrap(
        &self,
        irrefutable_facts: &crate::solver::OwnedPartialModel,
    ) -> Self::OutputUnwrap {
        let data = self
            .data
            .iter()
            .map(|v| irrefutable_facts.get_unwrap(v))
            .collect::<Vec<_>>();
        HexGrid::from_raw(self.dims(), data)
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct HexInnerGridEdges<T> {
    pub dims: (usize, usize, usize, usize),
    pub to_right: HexGrid<T>,
    pub to_bottom_left: HexGrid<T>,
    pub to_bottom_right: HexGrid<T>,
}

pub fn borders_to_rooms(borders: &HexInnerGridEdges<bool>) -> Vec<Vec<(usize, usize)>> {
    let cell_mapping = HexCellMapping::new(borders.dims);

    fn visit(
        y: usize,
        x: usize,
        cell_mapping: &HexCellMapping,
        borders: &HexInnerGridEdges<bool>,
        visited: &mut Vec<Vec<bool>>,
        room: &mut Vec<(usize, usize)>,
    ) {
        if !cell_mapping.is_valid_coord((y, x)) {
            return;
        }
        if visited[y][x] {
            return;
        }
        visited[y][x] = true;
        room.push((y, x));

        if cell_mapping.is_valid_coord_offset((y, x), (-1, 0))
            && !borders.to_bottom_left[(y - 1, x)]
        {
            visit(y - 1, x, cell_mapping, borders, visited, room);
        }
        if cell_mapping.is_valid_coord_offset((y, x), (1, 0)) && !borders.to_bottom_left[(y, x)] {
            visit(y + 1, x, cell_mapping, borders, visited, room);
        }
        if cell_mapping.is_valid_coord_offset((y, x), (0, -1)) && !borders.to_right[(y, x - 1)] {
            visit(y, x - 1, cell_mapping, borders, visited, room);
        }
        if cell_mapping.is_valid_coord_offset((y, x), (0, 1)) && !borders.to_right[(y, x)] {
            visit(y, x + 1, cell_mapping, borders, visited, room);
        }
        if cell_mapping.is_valid_coord_offset((y, x), (-1, -1))
            && !borders.to_bottom_right[(y - 1, x - 1)]
        {
            visit(y - 1, x - 1, cell_mapping, borders, visited, room);
        }
        if cell_mapping.is_valid_coord_offset((y, x), (1, 1)) && !borders.to_bottom_right[(y, x)] {
            visit(y + 1, x + 1, cell_mapping, borders, visited, room);
        }
    }

    let (rh, rw) = cell_mapping.repr_dims();
    let mut visited = vec![vec![false; rw]; rh];

    let mut rooms = vec![];
    for y in 0..rh {
        for x in 0..rw {
            if !cell_mapping.is_valid_coord((y, x)) {
                continue;
            }

            if visited[y][x] {
                continue;
            }

            let mut room = vec![];
            visit(y, x, &cell_mapping, borders, &mut visited, &mut room);
            rooms.push(room);
        }
    }

    rooms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_cell_mapping() {
        //      * * * * *
        //     * * * * * *
        //    * * * * * * *
        //   * * * * * * *
        //  * * * * * * *
        // * * * * * * *
        //  * * * * * *
        //   * * * * *
        //    * * * *
        assert_eq!(num_cells((6, 5, 4, 3)), 54);
    }

    #[test]
    fn test_borders_to_rooms() {
        let borders = HexInnerGridEdges {
            dims: (3, 3, 2, 2),
            to_right: HexGrid::from_grid(
                (3, 3, 2, 2),
                vec![
                    vec![true, false, false, false],
                    vec![true, true, true, false],
                    vec![true, true, true, false],
                    vec![false, false, true, false],
                ],
            ),
            to_bottom_left: HexGrid::from_grid(
                (3, 3, 2, 2),
                vec![
                    vec![false, false, true, false],
                    vec![true, true, true, true],
                    vec![false, true, true, false],
                    vec![false, false, false, false],
                ],
            ),
            to_bottom_right: HexGrid::from_grid(
                (3, 3, 2, 2),
                vec![
                    vec![true, true, false, false],
                    vec![false, false, true, false],
                    vec![true, true, true, false],
                    vec![false, false, false, false],
                ],
            ),
        };

        let mut actual = borders_to_rooms(&borders);
        let mut expected = vec![
            vec![(0, 0), (1, 0), (2, 1)],
            vec![(0, 1), (0, 2), (1, 1), (1, 3), (2, 2)],
            vec![(1, 2)],
            vec![(2, 0)],
            vec![(2, 3), (3, 3)],
            vec![(3, 1), (3, 2)],
        ];

        for n in [&mut actual, &mut expected] {
            for room in n.iter_mut() {
                room.sort();
            }

            n.sort();
        }

        assert_eq!(actual, expected);
    }
}
