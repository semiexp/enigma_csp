use cspuz_rs::graph::{GridEdges, InnerGridEdges};

pub enum Uniqueness {
    Unique,
    NonUnique,
    NotApplicable,
}

pub trait UniquenessCheckable {
    fn is_unique(&self) -> bool;
}

impl<T> UniquenessCheckable for Option<T> {
    fn is_unique(&self) -> bool {
        self.is_some()
    }
}

impl<T: UniquenessCheckable> UniquenessCheckable for Vec<T> {
    fn is_unique(&self) -> bool {
        self.iter().all(|x| x.is_unique())
    }
}

impl<U: UniquenessCheckable, V: UniquenessCheckable> UniquenessCheckable for (&U, &V) {
    fn is_unique(&self) -> bool {
        self.0.is_unique() && self.1.is_unique()
    }
}

impl<T: UniquenessCheckable> UniquenessCheckable for GridEdges<T> {
    fn is_unique(&self) -> bool {
        self.horizontal.is_unique() && self.vertical.is_unique()
    }
}

impl<T: UniquenessCheckable> UniquenessCheckable for InnerGridEdges<T> {
    fn is_unique(&self) -> bool {
        self.horizontal.is_unique() && self.vertical.is_unique()
    }
}

pub fn is_unique<T>(x: &T) -> Uniqueness
where
    T: UniquenessCheckable,
{
    if x.is_unique() {
        Uniqueness::Unique
    } else {
        Uniqueness::NonUnique
    }
}
