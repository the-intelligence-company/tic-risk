//! the context the entity attends to besides its own stream: the economy series at the vintage
//! visible on the day and the industry production series matched to the company's division or
//! section, as `context.economy` / `context.industry` records; the sector quartiles are read here
//! too (state.rs writes the comparison records). the series vintage rule is the harness's
//! (publication date, else period plus the lag of the frequency).

use std::collections::HashMap;
use std::path::Path;

use risk_eval::store::{Economy, Sector, Series};

use crate::Error;
use crate::dates::{Day, months_between};
use crate::fields::Unit;
use crate::record::{CellType, FLAG_EXTRA, PeerType, Record};
use crate::state::{Cells, ExtraBuf};
use crate::streams::Streams;
use crate::vocab::Vocab;

pub struct Context {
    pub economy: Economy,
    pub sector: Sector,
    /// field id per national series
    national_fields: Vec<u32>,
    /// (published label, field id) per industry series
    industry_fields: Vec<(String, u32)>,
    /// source coverage: the input's own `coverage.json` when present, else the constant table
    pub coverage: Vec<Coverage>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Coverage {
    pub source: String,
    pub live_from: Option<chrono::NaiveDate>,
    pub ended: Option<chrono::NaiveDate>,
}

impl Coverage {
    /// `Some(true)` live at `at`, `Some(false)` not yet or no longer, `None` never dated
    pub fn is_live(&self, at: chrono::NaiveDate) -> Option<bool> {
        let from = self.live_from?;
        let ended_before = self.ended.is_some_and(|e| at >= e);
        Some(at >= from && !ended_before)
    }
}

/// the constant table of source coverage (the sources' live and end dates)
pub fn constant_coverage() -> Vec<Coverage> {
    risk_store::coverage::SOURCE_COVERAGE
        .iter()
        .map(|c| Coverage {
            source: c.source.to_owned(),
            live_from: c.live_from_date(),
            ended: c.ended_date(),
        })
        .collect()
}

/// `coverage.json` of the input directory, else the constant table
pub fn load_coverage(risk_dir: &Path) -> Result<Vec<Coverage>, Error> {
    let path = risk_dir.join("coverage.json");
    if path.exists() {
        let rows: Vec<Coverage> = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        if rows.is_empty() {
            return Err(Error::Other(format!("{} is empty", path.display())));
        }
        return Ok(rows);
    }
    Ok(risk_store::coverage::SOURCE_COVERAGE
        .iter()
        .map(|c| Coverage {
            source: c.source.to_owned(),
            live_from: c.live_from_date(),
            ended: c.ended_date(),
        })
        .collect())
}

/// one series as seen on a day: (period, value, 12-month change)
pub type SeriesView = Option<(Day, f32, f32)>;

pub struct EcoView {
    pub national: Vec<SeriesView>,
    pub industry: Vec<SeriesView>,
}

fn series_view(s: &Series, t: Day) -> SeriesView {
    let (period, value) = s.at(t)?;
    let [_, change] = s.view(t);
    Some((period, value, change))
}

/// an industry series label (`08`, `B`, `10-12`, `10-12,19`) against a division and its section
pub fn label_matches(label: &str, division: u32, section: char) -> bool {
    label.split(',').map(str::trim).any(|part| {
        if part.len() == 1 {
            return part.starts_with(section);
        }
        if let Some((a, b)) = part.split_once('-')
            && let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>())
        {
            return (a..=b).contains(&division);
        }
        part.parse::<u32>().ok() == Some(division)
    })
}

impl Context {
    pub fn open(risk_dir: &Path, vocab: &Vocab) -> Result<Context, Error> {
        let economy = Economy::read(&risk_dir.join("economy.jsonl"))?;
        let sector = Sector::read(&risk_dir.join("sector.jsonl"))?;
        Context::from_parts(economy, sector, vocab, load_coverage(risk_dir)?)
    }

    /// the context over an economy and sector table already read; every series must have its
    /// field in `vocab`
    pub fn from_parts(economy: Economy, sector: Sector, vocab: &Vocab, coverage: Vec<Coverage>) -> Result<Context, Error> {
        let field = |id: &str| {
            vocab
                .field(&format!("series.{id}"))
                .ok_or_else(|| Error::Other(format!("vocabulary lacks series {id}: rebuild")))
        };
        let national_fields = economy
            .national
            .iter()
            .map(|s| field(&s.id))
            .collect::<Result<Vec<_>, _>>()?;
        let industry_fields = economy
            .industry
            .iter()
            .map(|(label, s)| Ok((label.clone(), field(&s.id)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Context {
            economy,
            sector,
            national_fields,
            industry_fields,
            coverage,
        })
    }

    /// every series as seen on `t`
    pub fn view(&self, t: Day) -> EcoView {
        EcoView {
            national: self
                .economy
                .national
                .iter()
                .map(|s| series_view(s, t))
                .collect(),
            industry: self
                .economy
                .industry
                .iter()
                .map(|(_, s)| series_view(s, t))
                .collect(),
        }
    }

    /// views for a set of days, computed once each
    pub fn views(&self, days: &[Day]) -> HashMap<Day, EcoView> {
        use rayon::prelude::*;
        let mut uniq: Vec<Day> = days.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        uniq.par_iter().map(|&d| (d, self.view(d))).collect()
    }

    /// the context records of one sample
    #[allow(clippy::too_many_arguments)]
    pub fn records(
        &self,
        s: &Streams,
        view: &EcoView,
        entity: i32,
        division: Option<u32>,
        section: Option<char>,
        t: Day,
        out: &mut ExtraBuf,
    ) {
        let ids = &s.ids;
        let base = Record {
            entity,
            kind: ids.context_economy,
            valid: t,
            known: t,
            end: t,
            cell_start: 0,
            cell_count: 0,
            source: ids.source_tic,
            flags: FLAG_EXTRA,
            peer_type: PeerType::None,
            peer_id: 0,
            group: 0,
        };
        let change_f = ids.field("economy.change_12m");
        let period_f = ids.field("economy.period");
        for (i, v) in view.national.iter().enumerate() {
            let Some((period, value, change)) = v else {
                continue;
            };
            let mut c = Cells::new();
            c.num(self.national_fields[i], Unit::Index, *value);
            if change.is_finite() {
                c.num(change_f, Unit::Index, *change);
            }
            c.push(
                period_f,
                CellType::Months,
                Unit::Months,
                months_between(*period, t) as f32,
            );
            out.push(base, &c.cells);
        }
        let (Some(division), Some(section)) = (division, section) else {
            return;
        };
        for (i, v) in view.industry.iter().enumerate() {
            let Some((period, value, change)) = v else {
                continue;
            };
            let (label, field) = &self.industry_fields[i];
            if !label_matches(label, division, section) {
                continue;
            }
            let mut c = Cells::new();
            c.num(*field, Unit::Index, *value);
            if change.is_finite() {
                c.num(change_f, Unit::Index, *change);
            }
            c.push(
                period_f,
                CellType::Months,
                Unit::Months,
                months_between(*period, t) as f32,
            );
            out.push(
                Record {
                    kind: ids.context_industry,
                    ..base
                },
                &c.cells,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn industry_labels() {
        assert!(label_matches("08", 8, 'B'));
        assert!(label_matches("B", 8, 'B'));
        assert!(label_matches("10-12", 11, 'C'));
        assert!(!label_matches("10-12", 13, 'C'));
        assert!(label_matches("10-12,19", 19, 'C'));
        assert!(!label_matches("C", 69, 'M'));
    }
}
