//! anchor sets: the (entity, as-of) rows samples are built for, with the label block for every
//! horizon. the protocol anchors take the harness's questions, splits and rows, so the backbone
//! is measured on the same rows as any baseline of the question; the pretraining anchors take
//! monthly points per entity over the whole history, subsampled by a stable hash, with a held-out
//! share of entities. labels come from the label set through the harness's `outcome`, the one
//! place that writes labels.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use rayon::prelude::*;
use risk_eval::points::{OUTCOME_SETTLE_DAYS, Plan, Row, Split, splitmix};
use risk_eval::question::Question;
use risk_eval::store::{read_label_events, read_lives};
use risk_store::labels::{DefaultEvent, LabelSet};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::dates::{add_months, from_naive, to_naive};
use crate::record::{
    EVENT_INCOMPLETE, HORIZONS, HorizonLabel, SAMPLE_CENSORED, SAMPLE_COMPETING, SAMPLE_POSITIVE,
    SAMPLES_MAGIC, SampleRow, read_all, write_all,
};
use crate::sampler::Budget;
use crate::texts;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorPlan {
    pub name: String,
    /// `protocol` or `pretrain`
    pub mode: String,
    pub question: Option<String>,
    pub horizon_months: Option<u32>,
    /// the harness plan of a protocol set, as written
    pub eval_plan: Option<serde_json::Value>,
    pub cut: NaiveDate,
    pub from: Option<NaiveDate>,
    pub to: Option<NaiveDate>,
    pub per_entity: Option<u32>,
    pub holdout_pct: Option<u32>,
    pub seed: u64,
    pub rows: u64,
    pub positives: u64,
    /// the sampler budget, once samples are built
    pub budget: Option<Budget>,
    /// pretraining: the harness keys whose test companies are left out
    #[serde(default)]
    pub exclude_test: Vec<PathBuf>,
    /// pretraining: the fine-tune's validation companies (one in k, `val_entity`) left out; 0 none
    #[serde(default)]
    pub exclude_val_k: u32,
}

/// the fine-tune's validation companies: one in k by a multiplicative hash of the id. the
/// training split used this exact hash; changing it moves companies between train and validation
pub fn val_entity(entity: i32, k: u32) -> bool {
    k > 0 && (i64::from(entity).wrapping_mul(2_654_435_761) & 0xFFFF_FFFF) % i64::from(k) == 0
}

/// the companies a set of harness keys holds out as test (`split` = test)
pub fn test_companies(paths: &[PathBuf]) -> Result<HashSet<i32>, Error> {
    let mut out = HashSet::new();
    for path in paths {
        let mut lines = BufReader::new(File::open(path)?).lines();
        let header = lines.next().transpose()?.unwrap_or_default();
        let cols: Vec<&str> = header.trim_end().split(',').collect();
        let col = |name: &str| cols.iter().position(|c| *c == name);
        let (Some(ci), Some(si)) = (col("company_id"), col("split")) else {
            return Err(Error::Other(format!("{} lacks company_id or split", path.display())));
        };
        for line in lines {
            let line = line?;
            let p: Vec<&str> = line.trim_end().split(',').collect();
            if p.get(si) == Some(&"test")
                && let Some(c) = p.get(ci).and_then(|v| v.parse::<i32>().ok())
            {
                out.insert(c);
            }
        }
    }
    Ok(out)
}

pub fn open_labels(risk_dir: &Path, form: &str) -> Result<LabelSet, Error> {
    let lives = read_lives(&risk_dir.join(format!("lives_{form}.csv")))?;
    let records = read_label_events(&risk_dir.join(format!("label_events_{form}.csv")))?;
    Ok(LabelSet::from_records(lives, records))
}

pub fn event_code(e: DefaultEvent) -> u8 {
    DefaultEvent::ALL
        .iter()
        .position(|x| *x == e)
        .map_or(0, |i| i as u8 + 1)
}

pub fn query_index(key: &str) -> u8 {
    texts::QUERIES
        .iter()
        .position(|(k, _, _)| *k == key)
        .and_then(|i| u8::try_from(i).ok())
        .unwrap_or(0)
}

pub fn question_query(q: &Question) -> u8 {
    query_index(q.target.code())
}

fn complete(as_of: NaiveDate, horizon: u32, cut: NaiveDate) -> bool {
    add_months(as_of, horizon)
        .checked_add_days(chrono::Days::new(OUTCOME_SETTLE_DAYS))
        .is_some_and(|end| end <= cut)
}

/// the first default-type event, days to it, censoring, completeness, per horizon
pub fn label_block(
    labels: &LabelSet,
    entity: i32,
    as_of: NaiveDate,
    cut: NaiveDate,
) -> [HorizonLabel; 4] {
    let mut out = [HorizonLabel::default(); 4];
    for (i, h) in HORIZONS.iter().enumerate() {
        if !complete(as_of, *h, cut) {
            out[i] = HorizonLabel {
                event: EVENT_INCOMPLETE,
                censored: 0,
                days: -1,
            };
            continue;
        }
        let Some(label) = labels.label_at(entity, as_of, *h) else {
            out[i] = HorizonLabel {
                event: EVENT_INCOMPLETE,
                censored: 0,
                days: -1,
            };
            continue;
        };
        out[i] = match (label.event, label.censored_at) {
            (Some((e, d)), _) => HorizonLabel {
                event: event_code(e),
                censored: 0,
                days: (d - as_of).num_days() as i32,
            },
            (None, Some(c)) => HorizonLabel {
                event: 0,
                censored: 1,
                days: (c - as_of).num_days() as i32,
            },
            (None, None) => HorizonLabel {
                event: 0,
                censored: 0,
                days: -1,
            },
        };
    }
    out
}

/// the harness's rows of a question and plan as anchors
pub fn protocol(labels: &LabelSet, q: &Question, plan: &Plan, rows: &[Row]) -> Vec<SampleRow> {
    let query = question_query(q);
    rows.par_iter()
        .enumerate()
        .map(|(i, r)| {
            let as_of = to_naive(r.as_of);
            let mut flags = 0u8;
            if r.outcome.positive {
                flags |= SAMPLE_POSITIVE;
            }
            if r.outcome.competing {
                flags |= SAMPLE_COMPETING;
            }
            if r.outcome.censored {
                flags |= SAMPLE_CENSORED;
            }
            SampleRow {
                entity: r.company,
                as_of: r.as_of,
                split: match r.split {
                    Split::Train => 0,
                    Split::Test => 1,
                },
                flags,
                query,
                horizon: u8::try_from(q.horizon_months).unwrap_or(u8::MAX),
                weight: r.weight,
                labels: label_block(labels, r.company, as_of, plan.cut),
                row: i as u32,
            }
        })
        .collect()
}

/// monthly points per entity over `[from, to]` while in the risk set, `per_entity` of them by a
/// stable hash, entities held out by `holdout_pct`; the companies in `exclude` and the validation
/// companies of `val_k` are not pretrained on at all
#[allow(clippy::too_many_arguments)]
pub fn pretrain(
    labels: &LabelSet,
    from: NaiveDate,
    to: NaiveDate,
    per_entity: u32,
    holdout_pct: u32,
    cut: NaiveDate,
    seed: u64,
    exclude: &HashSet<i32>,
    val_k: u32,
) -> Vec<SampleRow> {
    let lives: Vec<_> = labels
        .lives()
        .filter(|l| !exclude.contains(&l.company_id) && !val_entity(l.company_id, val_k))
        .cloned()
        .collect();
    let query = query_index("pretrain");
    let mut out: Vec<SampleRow> = lives
        .par_iter()
        .flat_map_iter(|life| {
            let entity = life.company_id;
            let start = life.registered.map_or(from, |r| r.max(from));
            let end = life.ceased.map_or(to, |c| c.min(to));
            let mut points: Vec<(u64, NaiveDate)> = LabelSet::monthly_points(start, end)
                .filter(|p| labels.in_risk_set(entity, *p))
                .map(|p| {
                    let h = splitmix((entity as u64) << 32 ^ (from_naive(p) as u32 as u64) ^ seed);
                    (h, p)
                })
                .collect();
            points.sort_unstable();
            points.truncate(per_entity as usize);
            let split =
                u8::from(splitmix(entity as u64 ^ seed ^ 0x5eed) % 100 < u64::from(holdout_pct));
            points.into_iter().map(move |(_, p)| SampleRow {
                entity,
                as_of: from_naive(p),
                split,
                flags: 0,
                query,
                horizon: 0,
                weight: 1.0,
                labels: label_block(labels, entity, p, cut),
                row: 0,
            })
        })
        .collect();
    out.par_sort_unstable_by_key(|r| (r.as_of, r.entity));
    for (i, r) in out.iter_mut().enumerate() {
        r.row = i as u32;
    }
    out
}

pub fn save(dir: &Path, plan: &AnchorPlan, rows: &[SampleRow]) -> Result<(), Error> {
    std::fs::create_dir_all(dir)?;
    write_all(&dir.join("samples.bin"), SAMPLES_MAGIC, rows)?;
    serde_json::to_writer_pretty(BufWriter::new(File::create(dir.join("plan.json"))?), plan)?;
    Ok(())
}

pub fn load(dir: &Path) -> Result<(AnchorPlan, Vec<SampleRow>), Error> {
    let plan: AnchorPlan =
        serde_json::from_reader(BufReader::new(File::open(dir.join("plan.json"))?))?;
    let rows = read_all(&dir.join("samples.bin"), SAMPLES_MAGIC)?;
    Ok((plan, rows))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use risk_store::known_at::KnownAt;
    use risk_store::labels::EventRecord;
    use risk_store::population::{CompanyLife, LegalForm};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn set() -> LabelSet {
        let life = |id, ceased| CompanyLife {
            company_id: id,
            legal_form: LegalForm::Aktiebolag,
            registered: Some(d(2010, 1, 1)),
            ceased,
        };
        let rec = |id, event, date| EventRecord {
            company_id: id,
            event,
            valid_time: date,
            known_at: KnownAt {
                date,
                estimated: false,
            },
            ended: None,
        };
        LabelSet::from_records(
            vec![life(1, None), life(2, None), life(3, Some(d(2025, 3, 1)))],
            vec![
                rec(1, DefaultEvent::Bankruptcy, d(2025, 6, 1)),
                rec(2, DefaultEvent::CompulsoryLiquidation, d(2025, 4, 1)),
            ],
        )
    }

    #[test]
    fn label_blocks_follow_the_horizons_and_the_cut() {
        let s = set();
        let b = label_block(&s, 1, d(2025, 1, 1), d(2026, 10, 9));
        assert_eq!(
            b[0],
            HorizonLabel {
                event: 0,
                censored: 0,
                days: -1
            },
            "3 m: nothing"
        );
        assert_eq!(b[1].event, 1, "6 m: bankruptcy");
        assert_eq!(b[1].days, 151);
        assert_eq!(b[2].event, 1);
        assert_eq!(b[3].event, EVENT_INCOMPLETE, "24 m not complete at the cut");
        let b = label_block(&s, 2, d(2025, 1, 1), d(2026, 10, 9));
        assert_eq!(b[1].event, 2, "compulsory liquidation");
        let b = label_block(&s, 3, d(2025, 1, 1), d(2026, 10, 9));
        assert_eq!(b[0].censored, 1);
        assert_eq!(b[0].days, 59);
        let b = label_block(&s, 1, d(2025, 1, 1), d(2025, 9, 1));
        assert_eq!(b[1].event, 1, "6 m complete: 2025-07-01 + 30 d ≤ cut");
        assert_eq!(b[2].event, EVENT_INCOMPLETE, "12 m not complete at the cut");
        assert_eq!(event_code(DefaultEvent::Deregistration), 4);
        assert_eq!(query_index("any_default"), 4);
    }

    #[test]
    fn val_entity_matches_the_trainer() {
        // the hash written out: ((e * 2654435761) & 0xFFFFFFFF) % 10 == 0
        for e in [12, 24, 26, 38] {
            assert!(val_entity(e, 10), "{e}");
        }
        for e in [1, 7, 10, 1_882_540, 1_882_639, 3_776_603] {
            assert!(!val_entity(e, 10), "{e}");
        }
        assert!(!val_entity(12, 0), "k 0: no validation companies");
    }

    #[test]
    fn pretrain_anchors_are_stable_and_in_the_risk_set() {
        let s = set();
        let none = HashSet::new();
        let a = pretrain(&s, d(2024, 1, 1), d(2025, 12, 1), 5, 50, d(2026, 10, 9), 7, &none, 0);
        let b = pretrain(&s, d(2024, 1, 1), d(2025, 12, 1), 5, 50, d(2026, 10, 9), 7, &none, 0);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x == y));
        assert!(a.iter().filter(|r| r.entity == 1).count() <= 5);
        assert!(
            a.iter()
                .filter(|r| r.entity == 3)
                .all(|r| to_naive(r.as_of) < d(2025, 3, 1))
        );
        assert!(
            a.iter()
                .filter(|r| r.entity == 1)
                .all(|r| to_naive(r.as_of) < d(2025, 6, 1))
        );
        let e1 = a.iter().find(|r| r.entity == 1).unwrap().split;
        assert!(a.iter().filter(|r| r.entity == 1).all(|r| r.split == e1));
        // a held-out test company is not pretrained on; the others keep their anchors
        let held: HashSet<i32> = [1].into();
        let c = pretrain(&s, d(2024, 1, 1), d(2025, 12, 1), 5, 50, d(2026, 10, 9), 7, &held, 0);
        assert!(c.iter().all(|r| r.entity != 1));
        assert_eq!(c.len(), a.iter().filter(|r| r.entity != 1).count());
    }
}
