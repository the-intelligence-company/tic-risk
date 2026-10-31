//! as-of points and splits. a point is complete when its whole horizon plus a settling period lies
//! before the data cut. the temporal split tests on the most recent complete points and trains on
//! points whose horizons close before the first test point (no outcome overlap); the entity split
//! keeps one window and holds out companies by a stable hash, for questions whose horizon leaves
//! no training points with the late source families live. negatives are sampled by a stable hash
//! of (company, point) with an inverse-probability weight; positives are always kept.

use chrono::NaiveDate;
use rayon::prelude::*;
use risk_store::labels::LabelSet;
use serde::Serialize;

use crate::dates::{Day, add_months, first_of_month, from_naive, sub_months, to_naive};
use crate::question::{Events, Outcome, Question, outcome};

/// registration lag of the outcome events
pub const OUTCOME_SETTLE_DAYS: u64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum SplitRule {
    Temporal { test_months: u32, train_months: u32 },
    Entity { window_months: u32, test_pct: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Split {
    Train,
    Test,
}

impl Split {
    pub fn code(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub question: String,
    pub horizon_months: u32,
    pub rule: SplitRule,
    pub cut: NaiveDate,
    pub settle_days: u64,
    pub last_complete: NaiveDate,
    pub train_points: Vec<NaiveDate>,
    pub test_points: Vec<NaiveDate>,
    pub negatives: f64,
    pub test_negatives: f64,
}

fn complete(point: NaiveDate, horizon: u32, cut: NaiveDate) -> bool {
    add_months(point, horizon)
        .checked_add_days(chrono::Days::new(OUTCOME_SETTLE_DAYS))
        .is_some_and(|end| end <= cut)
}

pub fn plan(
    q: &Question,
    rule: SplitRule,
    cut: NaiveDate,
    negatives: f64,
    test_negatives: f64,
) -> Plan {
    let mut last = first_of_month(cut);
    while !complete(last, q.horizon_months, cut) {
        last = sub_months(last, 1);
    }
    let back = |end: NaiveDate, n: u32| -> Vec<NaiveDate> {
        (0..n).rev().map(|i| sub_months(end, i)).collect()
    };
    let (train_points, test_points) = match rule {
        SplitRule::Temporal {
            test_months,
            train_months,
        } => {
            let test = back(last, test_months);
            let first_test = test.first().copied().unwrap_or(last);
            // the last train point whose horizon closes at or before the first test point
            let last_train = sub_months(first_test, q.horizon_months);
            (back(last_train, train_months), test)
        }
        SplitRule::Entity { window_months, .. } => {
            let w = back(last, window_months);
            (w.clone(), w)
        }
    };
    Plan {
        question: q.name.to_owned(),
        horizon_months: q.horizon_months,
        rule,
        cut,
        settle_days: OUTCOME_SETTLE_DAYS,
        last_complete: last,
        train_points,
        test_points,
        negatives,
        test_negatives,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Row {
    pub company: i32,
    pub as_of: Day,
    pub split: Split,
    pub outcome: Outcome,
    /// inverse sampling probability (1 for positives)
    pub weight: f32,
}

pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn in_sample(company: i32, as_of: Day, rate: f64) -> bool {
    if rate >= 1.0 {
        return true;
    }
    let h = splitmix((company as u64) << 32 ^ (as_of as u32 as u64));
    ((h % 1_000_000) as f64) < rate * 1_000_000.0
}

/// the company's side under an entity split: stable across questions and runs
pub fn entity_split(company: i32, test_pct: u32) -> Split {
    if splitmix(company as u64 ^ 0x5eed) % 100 < u64::from(test_pct) {
        Split::Test
    } else {
        Split::Train
    }
}

/// every (company, point) in the risk set, labelled, split and sampled
pub fn rows(labels: &LabelSet, events: Option<&Events>, plan: &Plan, q: &Question) -> Vec<Row> {
    let points: Vec<(NaiveDate, Split)> = match plan.rule {
        SplitRule::Temporal { .. } => plan
            .train_points
            .iter()
            .map(|p| (*p, Split::Train))
            .chain(plan.test_points.iter().map(|p| (*p, Split::Test)))
            .collect(),
        SplitRule::Entity { .. } => plan
            .test_points
            .iter()
            .map(|p| (*p, Split::Train))
            .collect(),
    };
    let lives: Vec<i32> = labels.lives().map(|l| l.company_id).collect();
    let mut out: Vec<Row> = points
        .par_iter()
        .flat_map_iter(|&(point, point_split)| {
            let day = from_naive(point);
            lives.iter().filter_map(move |&company| {
                let o = outcome(labels, events, company, point, q)?;
                let split = match plan.rule {
                    SplitRule::Temporal { .. } => point_split,
                    SplitRule::Entity { test_pct, .. } => entity_split(company, test_pct),
                };
                let rate = match split {
                    Split::Train => plan.negatives,
                    Split::Test => plan.test_negatives,
                };
                if !o.positive && !in_sample(company, day, rate) {
                    return None;
                }
                Some(Row {
                    company,
                    as_of: day,
                    split,
                    outcome: o,
                    weight: if o.positive {
                        1.0
                    } else {
                        (1.0 / rate.max(1e-9)) as f32
                    },
                })
            })
        })
        .collect();
    out.par_sort_unstable_by_key(|r| (r.as_of, r.company));
    out
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct PointCount {
    pub as_of: NaiveDate,
    pub split: &'static str,
    pub at_risk: usize,
    pub positives: usize,
    pub competing: usize,
    pub censored: usize,
    pub rows: usize,
}

/// unsampled counts per point (the risk set and its outcomes) and the sampled row counts
pub fn point_counts(
    labels: &LabelSet,
    events: Option<&Events>,
    plan: &Plan,
    q: &Question,
    rows: &[Row],
) -> Vec<PointCount> {
    let points: Vec<(NaiveDate, &'static str)> = match plan.rule {
        SplitRule::Temporal { .. } => plan
            .train_points
            .iter()
            .map(|p| (*p, "train"))
            .chain(plan.test_points.iter().map(|p| (*p, "test")))
            .collect(),
        SplitRule::Entity { .. } => plan.test_points.iter().map(|p| (*p, "both")).collect(),
    };
    let lives: Vec<i32> = labels.lives().map(|l| l.company_id).collect();
    points
        .par_iter()
        .map(|&(point, split)| {
            let mut c = PointCount {
                as_of: point,
                split,
                ..Default::default()
            };
            for &company in &lives {
                if let Some(o) = outcome(labels, events, company, point, q) {
                    c.at_risk += 1;
                    c.positives += usize::from(o.positive);
                    c.competing += usize::from(o.competing);
                    c.censored += usize::from(o.censored);
                }
            }
            let day = from_naive(point);
            c.rows = rows.iter().filter(|r| r.as_of == day).count();
            c
        })
        .collect()
}

pub fn point_date(day: Day) -> NaiveDate {
    to_naive(day)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::question::find;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn temporal_plan_has_no_outcome_overlap() {
        let q = find("bankruptcy_12m").unwrap();
        let p = plan(
            q,
            SplitRule::Temporal {
                test_months: 12,
                train_months: 24,
            },
            d(2026, 10, 9),
            0.05,
            0.2,
        );
        // 2025-09-01 + 12 m + 30 d = 2026-10-01 ≤ cut; 2025-10-01 would end 2026-10-31
        assert_eq!(p.last_complete, d(2025, 9, 1));
        assert_eq!(p.test_points.first(), Some(&d(2024, 10, 1)));
        assert_eq!(p.test_points.len(), 12);
        assert_eq!(p.train_points.last(), Some(&d(2023, 10, 1)));
        assert_eq!(p.train_points.len(), 24);
        assert_eq!(p.train_points.first(), Some(&d(2021, 11, 1)));
        let q3 = find("bankruptcy_3m").unwrap();
        let p3 = plan(q3, p.rule, d(2026, 10, 9), 1.0, 1.0);
        assert_eq!(p3.last_complete, d(2026, 6, 1));
        assert_eq!(p3.train_points.last(), Some(&d(2025, 4, 1)));
    }

    #[test]
    fn entity_plan_shares_the_window_and_hashes_companies() {
        let q = find("bankruptcy_12m").unwrap();
        let p = plan(
            q,
            SplitRule::Entity {
                window_months: 6,
                test_pct: 10,
            },
            d(2026, 10, 9),
            1.0,
            1.0,
        );
        assert_eq!(p.train_points, p.test_points);
        assert_eq!(p.test_points.len(), 6);
        let test = (1..100_000)
            .filter(|c| entity_split(*c, 10) == Split::Test)
            .count();
        assert!((9_000..11_000).contains(&test), "{test}");
        assert_eq!(entity_split(4242, 10), entity_split(4242, 10));
    }

    #[test]
    fn sampling_is_stable_and_weighted() {
        let kept = (1..200_000).filter(|c| in_sample(*c, 20_000, 0.05)).count();
        assert!((9_000..11_000).contains(&kept), "{kept}");
        assert!(in_sample(5, 1, 1.0));
        assert_eq!(in_sample(77, 20_000, 0.05), in_sample(77, 20_000, 0.05));
    }
}
