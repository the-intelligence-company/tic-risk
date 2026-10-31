//! a fact's knowledge time: the day it became known, and whether that day is an estimate

use chrono::NaiveDate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownAt {
    pub date: NaiveDate,
    /// true when the day is derived (the fact's valid time plus a lag), not recorded
    pub estimated: bool,
}

impl KnownAt {
    /// never before `floor`: a record known earlier moves to it and is flagged estimated
    pub fn not_before(self, floor: NaiveDate) -> KnownAt {
        if self.date < floor {
            KnownAt {
                date: floor,
                estimated: true,
            }
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn a_floor_moves_only_what_is_earlier() {
        let k = KnownAt {
            date: d(2019, 3, 2),
            estimated: false,
        };
        assert_eq!(
            k.not_before(d(2021, 6, 1)),
            KnownAt {
                date: d(2021, 6, 1),
                estimated: true
            }
        );
        let k = KnownAt {
            date: d(2022, 2, 3),
            estimated: false,
        };
        assert_eq!(k.not_before(d(2021, 6, 1)), k);
    }
}
