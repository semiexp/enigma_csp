use crate::arithmetic::CheckedInt;
use crate::util::UpdateStatus;
use std::ops::{Add, BitOr, Mul};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Domain {
    Range(CheckedInt, CheckedInt),
    Enumerative(Vec<CheckedInt>),
}

impl Domain {
    pub fn range(low: i32, high: i32) -> Domain {
        Domain::Range(CheckedInt::new(low), CheckedInt::new(high))
    }

    pub fn enumerative(cands: Vec<i32>) -> Domain {
        Domain::Enumerative(cands.into_iter().map(CheckedInt::new).collect())
    }

    pub fn empty() -> Domain {
        Domain::range(1, 0)
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Domain::Range(low, high) => low > high,
            Domain::Enumerative(cands) => cands.is_empty(),
        }
    }

    pub(crate) fn range_from_checked(low: CheckedInt, high: CheckedInt) -> Domain {
        Domain::Range(low, high)
    }

    pub(crate) fn enumerative_from_checked(cands: Vec<CheckedInt>) -> Domain {
        Domain::Enumerative(cands)
    }

    pub(crate) fn enumerate(&self) -> Vec<CheckedInt> {
        match self {
            Domain::Range(low, high) => (low.get()..=high.get()).map(CheckedInt::new).collect(),
            Domain::Enumerative(cands) => cands.clone(),
        }
    }

    pub(crate) fn num_candidates(&self) -> usize {
        match self {
            &Domain::Range(low, high) => {
                if low <= high {
                    (high - low).get() as usize + 1
                } else {
                    0
                }
            }
            Domain::Enumerative(cands) => cands.len(),
        }
    }

    pub(crate) fn lower_bound_checked(&self) -> CheckedInt {
        match self {
            Domain::Range(low, _) => *low,
            Domain::Enumerative(cands) => {
                if cands.is_empty() {
                    CheckedInt::new(1)
                } else {
                    cands[0]
                }
            }
        }
    }

    pub(crate) fn upper_bound_checked(&self) -> CheckedInt {
        match self {
            Domain::Range(_, high) => *high,
            Domain::Enumerative(cands) => {
                if cands.is_empty() {
                    CheckedInt::new(0)
                } else {
                    cands[cands.len() - 1]
                }
            }
        }
    }

    pub(crate) fn as_constant(&self) -> Option<CheckedInt> {
        match self {
            Domain::Range(low, high) => {
                if *low == *high {
                    Some(*low)
                } else {
                    None
                }
            }
            Domain::Enumerative(cands) => {
                if cands.len() == 1 {
                    Some(cands[0])
                } else {
                    None
                }
            }
        }
    }

    pub fn is_infeasible(&self) -> bool {
        match self {
            Domain::Range(low, high) => *low > *high,
            Domain::Enumerative(cands) => cands.is_empty(),
        }
    }

    pub(crate) fn refine_upper_bound(&mut self, v: CheckedInt) -> UpdateStatus {
        match self {
            Domain::Range(low, high) => {
                if *high <= v {
                    UpdateStatus::NotUpdated
                } else {
                    *high = v;
                    if *low > *high {
                        UpdateStatus::Unsatisfiable
                    } else {
                        UpdateStatus::Updated
                    }
                }
            }
            Domain::Enumerative(cands) => {
                if cands.len() == 0 || cands[cands.len() - 1] <= v {
                    UpdateStatus::NotUpdated
                } else {
                    while !cands.is_empty() && cands[cands.len() - 1] > v {
                        cands.pop();
                    }
                    if cands.is_empty() {
                        UpdateStatus::Unsatisfiable
                    } else {
                        UpdateStatus::Updated
                    }
                }
            }
        }
    }

    pub(crate) fn refine_lower_bound(&mut self, v: CheckedInt) -> UpdateStatus {
        match self {
            Domain::Range(low, high) => {
                if *low >= v {
                    UpdateStatus::NotUpdated
                } else {
                    *low = v;
                    if *low > *high {
                        UpdateStatus::Unsatisfiable
                    } else {
                        UpdateStatus::Updated
                    }
                }
            }
            Domain::Enumerative(cands) => {
                if cands.len() == 0 || cands[0] >= v {
                    UpdateStatus::NotUpdated
                } else {
                    let mut start = 0;
                    while start < cands.len() && cands[start] < v {
                        start += 1;
                    }
                    let n_rem = cands.len() - start;
                    cands.rotate_right(n_rem);
                    cands.truncate(n_rem);

                    if n_rem == 0 {
                        UpdateStatus::Unsatisfiable
                    } else {
                        UpdateStatus::Updated
                    }
                }
            }
        }
    }
}

impl Add<Domain> for Domain {
    type Output = Domain;

    fn add(self, rhs: Domain) -> Domain {
        // TODO: reduce candidates for enumerative domains
        if self.is_infeasible() || rhs.is_infeasible() {
            return Domain::empty();
        }

        let low1 = self.lower_bound_checked();
        let high1 = self.upper_bound_checked();
        let low2 = rhs.lower_bound_checked();
        let high2 = rhs.upper_bound_checked();
        Domain::Range(low1 + low2, high1 + high2)
    }
}

impl Mul<CheckedInt> for Domain {
    type Output = Domain;

    fn mul(self, rhs: CheckedInt) -> Domain {
        match self {
            Domain::Range(low, high) => {
                if rhs == 0 {
                    if low > high {
                        Domain::Range(low, high)
                    } else {
                        Domain::range(0, 0)
                    }
                } else if rhs > 0 {
                    Domain::Range(low * rhs, high * rhs)
                } else {
                    Domain::Range(high * rhs, low * rhs)
                }
            }
            Domain::Enumerative(mut cands) => {
                if rhs == 0 {
                    if cands.len() == 0 {
                        Domain::empty()
                    } else {
                        Domain::range(0, 0)
                    }
                } else if rhs > 0 {
                    cands.iter_mut().for_each(|x| *x *= rhs);
                    Domain::Enumerative(cands)
                } else {
                    cands.iter_mut().for_each(|x| *x *= rhs);
                    cands.reverse();
                    Domain::Enumerative(cands)
                }
            }
        }
    }
}

impl BitOr<Domain> for Domain {
    type Output = Domain;

    fn bitor(self, rhs: Domain) -> Domain {
        // TODO: reduce candidates for enumerative domains
        if self.is_infeasible() {
            rhs
        } else if rhs.is_infeasible() {
            self
        } else {
            let low1 = self.lower_bound_checked();
            let high1 = self.upper_bound_checked();
            let low2 = rhs.lower_bound_checked();
            let high2 = rhs.upper_bound_checked();

            Domain::Range(low1.min(low2), high1.max(high2))
        }
    }
}
