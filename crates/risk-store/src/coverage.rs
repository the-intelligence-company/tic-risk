//! source coverage: the day each source's records start counting, as a constant list. the as-of
//! view answers *unknown* before a source's start, never "no event". a `None` start means a static
//! attribute or a source that is always available; an `ended` means the source stopped.

use chrono::NaiveDate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceCoverage {
    pub source: &'static str,
    /// (year, month, day); None = always available
    pub live_from: Option<(i32, u32, u32)>,
    pub ended: Option<(i32, u32, u32)>,
}

impl SourceCoverage {
    pub fn live_from_date(&self) -> Option<NaiveDate> {
        self.live_from
            .and_then(|(y, m, d)| NaiveDate::from_ymd_opt(y, m, d))
    }

    pub fn ended_date(&self) -> Option<NaiveDate> {
        self.ended
            .and_then(|(y, m, d)| NaiveDate::from_ymd_opt(y, m, d))
    }

    /// `Some(true)` live at `at`, `Some(false)` not yet or no longer, `None` never dated
    pub fn is_live(&self, at: NaiveDate) -> Option<bool> {
        let from = self.live_from_date()?;
        let ended_before = self.ended_date().is_some_and(|e| at >= e);
        Some(at >= from && !ended_before)
    }
}

const fn src(source: &'static str, live_from: Option<(i32, u32, u32)>) -> SourceCoverage {
    SourceCoverage {
        source,
        live_from,
        ended: None,
    }
}

pub const SOURCE_COVERAGE: &[SourceCoverage] = &[
    src("economy_series", Some((1976, 1, 1))),
    src("bankruptcies", Some((1990, 1, 1))),
    src("statuses_liquidations_reconstructions", Some((1990, 1, 1))),
    src("registered_case_list", Some((1991, 1, 1))),
    src("purpose_text", None),
    src("official_bankruptcy_statistics", Some((1996, 1, 1))),
    src("beneficial_owners", Some((2017, 1, 1))),
    src("industry_key_figures", Some((2019, 1, 1))),
    src("roles", Some((2020, 1, 1))),
    src("payroll_declarations", Some((2022, 2, 1))),
    src("annual_report_registrations", Some((2024, 1, 1))),
    src("scb_sni", Some((2025, 8, 20))),
    src("skatteverket_penalties", Some((2024, 1, 1))),
    src("bolagsverket_notifications", Some((2024, 11, 22))),
    src("court_documents", Some((2024, 11, 1))),
    src("kfm_change_files", Some((2024, 12, 11))),
    src("payment_orders", Some((2024, 12, 11))),
    src("kfm_snapshots", Some((2024, 12, 13))),
    src("late_fees_complete", Some((2025, 1, 1))),
    src("fskatt_register", Some((2025, 3, 1))),
    src("police_vulnerable_areas", Some((2026, 2, 1))),
    SourceCoverage {
        source: "representative_history",
        live_from: Some((2020, 1, 1)),
        ended: Some((2025, 6, 1)),
    },
    SourceCoverage {
        source: "composition_ackord",
        live_from: Some((1990, 1, 1)),
        ended: Some((1997, 1, 1)),
    },
];

pub fn find(source: &str) -> Option<&'static SourceCoverage> {
    SOURCE_COVERAGE.iter().find(|s| s.source == source)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn unknown_before_live_and_after_end() {
        let kfm = find("kfm_snapshots").unwrap();
        assert_eq!(kfm.is_live(d(2024, 12, 12)), Some(false));
        assert_eq!(kfm.is_live(d(2024, 12, 13)), Some(true));
        let rep = find("representative_history").unwrap();
        assert_eq!(rep.is_live(d(2024, 1, 1)), Some(true));
        assert_eq!(rep.is_live(d(2025, 6, 1)), Some(false));
        assert_eq!(find("purpose_text").unwrap().is_live(d(2000, 1, 1)), None);
        assert_eq!(find("payment_orders").unwrap().is_live(d(2024, 12, 10)), Some(false));
        assert_eq!(find("payroll_declarations").unwrap().is_live(d(2022, 2, 1)), Some(true));
    }

    #[test]
    fn every_source_named_once() {
        let mut names: Vec<_> = SOURCE_COVERAGE.iter().map(|s| s.source).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SOURCE_COVERAGE.len());
    }
}
