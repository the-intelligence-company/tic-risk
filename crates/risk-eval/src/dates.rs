//! day numbers: days since 1970-01-01 as i32. days before 1970 are negative, and the far
//! sentinels mark open intervals.

use chrono::{Datelike, Days, Months, NaiveDate};

pub type Day = i32;

/// an interval start that was never given
pub const DAY_MIN: Day = i32::MIN / 2;
/// an interval end that was never given (still open)
pub const DAY_MAX: Day = i32::MAX / 2;

fn epoch() -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or_default()
}

pub fn from_naive(d: NaiveDate) -> Day {
    i32::try_from((d - epoch()).num_days()).unwrap_or(DAY_MAX)
}

pub fn to_naive(d: Day) -> NaiveDate {
    epoch()
        .checked_add_signed(chrono::Duration::days(i64::from(d)))
        .unwrap_or_default()
}

/// `YYYY-MM-DD`; anything else is None
pub fn parse_ymd(s: &str) -> Option<Day> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u32> {
        let mut v = 0u32;
        for &c in &b[r] {
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + u32::from(c - b'0');
        }
        Some(v)
    };
    let y = i32::try_from(num(0..4)?).ok()?;
    NaiveDate::from_ymd_opt(y, num(5..7)?, num(8..10)?).map(from_naive)
}

pub fn fmt(d: Day) -> String {
    to_naive(d).format("%Y-%m-%d").to_string()
}

pub fn first_of_month(d: NaiveDate) -> NaiveDate {
    d.checked_sub_days(Days::new(u64::from(d.day0())))
        .unwrap_or(d)
}

pub fn add_months(d: NaiveDate, months: u32) -> NaiveDate {
    d.checked_add_months(Months::new(months)).unwrap_or(d)
}

pub fn sub_months(d: NaiveDate, months: u32) -> NaiveDate {
    d.checked_sub_months(Months::new(months)).unwrap_or(d)
}

/// whole months between two days (floor), for ages
pub fn months_between(from: Day, to: Day) -> i32 {
    let a = to_naive(from);
    let b = to_naive(to);
    let m = (b.year() - a.year()) * 12
        + (i32::try_from(b.month()).unwrap_or(0) - i32::try_from(a.month()).unwrap_or(0));
    if b.day() < a.day() { m - 1 } else { m }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn round_trips_and_parses() {
        let d = NaiveDate::from_ymd_opt(2024, 12, 2).unwrap();
        assert_eq!(to_naive(from_naive(d)), d);
        assert_eq!(parse_ymd("2024-12-02"), Some(from_naive(d)));
        assert_eq!(
            parse_ymd("1947-03-01"),
            Some(from_naive(NaiveDate::from_ymd_opt(1947, 3, 1).unwrap()))
        );
        assert_eq!(parse_ymd("2024-13-02"), None);
        assert_eq!(parse_ymd("2024-12-2"), None);
        assert_eq!(fmt(parse_ymd("1990-01-01").unwrap()), "1990-01-01");
    }

    #[test]
    fn month_arithmetic() {
        let d = NaiveDate::from_ymd_opt(2025, 1, 31).unwrap();
        assert_eq!(
            first_of_month(d),
            NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()
        );
        assert_eq!(
            add_months(d, 1),
            NaiveDate::from_ymd_opt(2025, 2, 28).unwrap()
        );
        assert_eq!(
            months_between(
                from_naive(NaiveDate::from_ymd_opt(2020, 6, 15).unwrap()),
                from_naive(d)
            ),
            55
        );
        assert_eq!(
            months_between(
                from_naive(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
                from_naive(d)
            ),
            12
        );
    }
}
