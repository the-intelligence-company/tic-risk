//! the in-memory store over the record cache: the cached events and relations with their
//! per-company ranges, the reverse indexes of a two-step walk (companies at an address, children
//! of a parent), the distress dates per company (a neighbour's distress as of a day), the label
//! set, and the two context files (economy series with vintages, sector key figures by industry
//! code and year).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chrono::NaiveDate;
use risk_store::known_at::KnownAt;
use risk_store::labels::{DefaultEvent, EventRecord, LabelSet};
use risk_store::population::{CompanyLife, LegalForm};
use serde::Deserialize;

use crate::Error;
use crate::cache::{self, Dict, EndpointType, EventRec, RelRec};
use crate::dates::{Day, from_naive, parse_ymd};

/// event kinds whose occurrence marks a neighbour as distressed
pub const DISTRESS_KINDS: &[&str] = &[
    "bankruptcy.initiated",
    "reconstruction.decided",
    "liquidation_order.issued",
    "debt.registered",
    "debt.present",
    "enforcement.case_registered",
    "fee.registered",
    "status.struck_off",
];

pub struct Kinds {
    pub industry_code: Option<u16>,
    pub group_parent: Option<u16>,
    pub ownership_via: Option<u16>,
    pub address: Option<u16>,
    pub liquidation_decided: Option<u16>,
    pub compulsory_true: Option<u16>,
    pub report_figures: Option<u16>,
    pub distress: Vec<u16>,
}

pub struct Store {
    pub dict: Dict,
    pub events: Vec<EventRec>,
    pub rels: Vec<RelRec>,
    ev_index: HashMap<i32, (u32, u32)>,
    rel_index: HashMap<i32, (u32, u32)>,
    by_address: HashMap<i64, Vec<u32>>,
    children: HashMap<i32, Vec<u32>>,
    /// (valid, known) of distress events per company, in valid order
    distress: HashMap<i32, Vec<(Day, Day)>>,
    pub labels: LabelSet,
    pub economy: Economy,
    pub sector: Sector,
    pub kinds: Kinds,
    /// `figures.bin`: one row per `report.figures` event, `dict.figure_names` order
    figures: Vec<f32>,
    pub figure_width: usize,
}

fn ranges<T>(items: &[T], key: impl Fn(&T) -> i32) -> HashMap<i32, (u32, u32)> {
    let mut idx = HashMap::new();
    let mut start = 0usize;
    while start < items.len() {
        let k = key(&items[start]);
        let mut end = start + 1;
        while end < items.len() && key(&items[end]) == k {
            end += 1;
        }
        idx.insert(k, (start as u32, end as u32));
        start = end;
    }
    idx
}

fn field<'a>(parts: &[&'a str], i: usize, line: u64) -> Result<&'a str, Error> {
    parts.get(i).copied().ok_or_else(|| Error::Record {
        line,
        msg: format!("missing field {i}"),
    })
}

fn opt_date(s: &str) -> Option<NaiveDate> {
    if s.is_empty() {
        None
    } else {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
    }
}

pub fn read_lives(path: &Path) -> Result<Vec<CompanyLife>, Error> {
    let mut out = Vec::new();
    for (i, line) in BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .skip(1)
    {
        let line = line?;
        let p: Vec<&str> = line.trim_end().split(',').collect();
        let no = i as u64 + 1;
        out.push(CompanyLife {
            company_id: field(&p, 0, no)?.parse().map_err(|_| Error::Record {
                line: no,
                msg: "company_id".into(),
            })?,
            legal_form: LegalForm::from_code(field(&p, 1, no)?).unwrap_or(LegalForm::Aktiebolag),
            registered: opt_date(field(&p, 2, no)?),
            ceased: opt_date(field(&p, 3, no)?),
        });
    }
    Ok(out)
}

pub fn read_label_events(path: &Path) -> Result<Vec<EventRecord>, Error> {
    let mut out = Vec::new();
    for (i, line) in BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .skip(1)
    {
        let line = line?;
        let p: Vec<&str> = line.trim_end().split(',').collect();
        let no = i as u64 + 1;
        let bad = |msg: &str| Error::Record {
            line: no,
            msg: msg.into(),
        };
        out.push(EventRecord {
            company_id: field(&p, 0, no)?.parse().map_err(|_| bad("company_id"))?,
            event: DefaultEvent::from_code(field(&p, 1, no)?).ok_or_else(|| bad("event"))?,
            valid_time: opt_date(field(&p, 2, no)?).ok_or_else(|| bad("valid_time"))?,
            known_at: KnownAt {
                date: opt_date(field(&p, 3, no)?).ok_or_else(|| bad("known_at"))?,
                estimated: field(&p, 4, no)? == "true",
            },
            ended: opt_date(field(&p, 5, no)?),
        });
    }
    Ok(out)
}

impl Store {
    pub fn open(risk_dir: &Path, form: &str) -> Result<Store, Error> {
        let cache_dir = risk_dir.join("cache");
        let dict = cache::load_dict(&cache_dir)?;
        let events = cache::load_events(&cache_dir)?;
        let rels = cache::load_relations(&cache_dir)?;
        let (figure_width, figures) = cache::load_figures(&cache_dir)?;
        tracing::info!(
            events = events.len(),
            relations = rels.len(),
            "cache loaded"
        );
        let lives = read_lives(&risk_dir.join(format!("lives_{form}.csv")))?;
        let records = read_label_events(&risk_dir.join(format!("label_events_{form}.csv")))?;
        let labels = LabelSet::from_records(lives, records);
        let economy = Economy::read(&risk_dir.join("economy.jsonl"))?;
        let sector = Sector::read(&risk_dir.join("sector.jsonl"))?;

        let kinds = Kinds {
            industry_code: dict.event_kind("industry.code"),
            group_parent: dict.rel_kind("group.parent"),
            ownership_via: dict.rel_kind("ownership.beneficial_via"),
            address: dict.rel_kind("address.registered"),
            liquidation_decided: dict.event_kind("liquidation.decided"),
            compulsory_true: dict.cat("compulsory=true"),
            report_figures: dict.event_kind("report.figures"),
            distress: DISTRESS_KINDS
                .iter()
                .filter_map(|k| dict.event_kind(k))
                .collect(),
        };
        let ev_index = ranges(&events, |e| e.company);
        let rel_index = ranges(&rels, |r| r.company);
        let mut by_address: HashMap<i64, Vec<u32>> = HashMap::new();
        let mut children: HashMap<i32, Vec<u32>> = HashMap::new();
        for (i, r) in rels.iter().enumerate() {
            if r.b_type == EndpointType::Address {
                by_address.entry(r.b_id).or_default().push(i as u32);
            } else if Some(r.kind) == kinds.group_parent
                && r.b_type == EndpointType::Company
                && let Ok(parent) = i32::try_from(r.b_id)
            {
                children.entry(parent).or_default().push(i as u32);
            }
        }
        let mut distress: HashMap<i32, Vec<(Day, Day)>> = HashMap::new();
        for e in &events {
            let is_distress = kinds.distress.contains(&e.kind)
                || (Some(e.kind) == kinds.liquidation_decided
                    && Some(e.cat2) == kinds.compulsory_true);
            if is_distress {
                distress
                    .entry(e.company)
                    .or_default()
                    .push((e.valid, e.known));
            }
        }
        Ok(Store {
            dict,
            events,
            rels,
            ev_index,
            rel_index,
            by_address,
            children,
            distress,
            labels,
            economy,
            sector,
            kinds,
            figures,
            figure_width,
        })
    }

    /// the figures row of a `report.figures` event
    pub fn figures_of(&self, e: &EventRec) -> Option<&[f32]> {
        if e.extra == cache::NO_EXTRA || self.figure_width == 0 {
            return None;
        }
        let start = e.extra as usize * self.figure_width;
        self.figures.get(start..start + self.figure_width)
    }

    /// the column of a figure in `figures_of`
    pub fn figure(&self, name: &str) -> Option<usize> {
        self.dict.figure_names.iter().position(|n| n == name)
    }

    pub fn events_of(&self, company: i32) -> &[EventRec] {
        self.ev_index
            .get(&company)
            .map_or(&[], |&(a, b)| &self.events[a as usize..b as usize])
    }

    pub fn rels_of(&self, company: i32) -> &[RelRec] {
        self.rel_index
            .get(&company)
            .map_or(&[], |&(a, b)| &self.rels[a as usize..b as usize])
    }

    /// indices into `rels` of the address relations at this address
    pub fn at_address(&self, address: i64) -> &[u32] {
        self.by_address.get(&address).map_or(&[], Vec::as_slice)
    }

    /// indices into `rels` of the `group.parent` edges whose parent is `company`
    pub fn children_of(&self, company: i32) -> &[u32] {
        self.children.get(&company).map_or(&[], Vec::as_slice)
    }

    /// a distress event of `company` with valid time in `(as_of - days, as_of]`, known by `as_of`
    pub fn distressed_within(&self, company: i32, as_of: Day, days: i32) -> bool {
        self.distress.get(&company).is_some_and(|v| {
            v.iter()
                .any(|&(valid, known)| known <= as_of && valid <= as_of && valid > as_of - days)
        })
    }

    pub fn companies(&self) -> usize {
        self.labels.lives().count()
    }
}

#[derive(Deserialize)]
struct EconomyLine {
    series: String,
    name: String,
    freq: String,
    sni: Option<String>,
    region: Option<String>,
    period: String,
    value: Option<f64>,
    published: String,
}

/// a series without vintages: a value is taken as known at the period's start plus the
/// publication lag of its frequency (monthly statistics a month and a half after the month,
/// quarterly accounts two months after the quarter), the estimate rule for undated sources.
/// daily rates are published on the day and keep their date
pub fn publication_lag_days(freq: &str) -> i32 {
    match freq {
        "D" => 1,
        "M" => 31 + 45,
        "Q" => 92 + 60,
        _ => 365 + 60,
    }
}

/// (period, published, value)
type Rows = Vec<(Day, Day, f32)>;

/// one economy series in row form: its description and its (period, value, published) rows
#[derive(Debug, Clone)]
pub struct SeriesRows {
    pub key: String,
    pub name: String,
    pub freq: Option<String>,
    pub sni: Option<String>,
    pub region: Option<String>,
    pub rows: Vec<(NaiveDate, Option<f64>, NaiveDate)>,
}

#[derive(Clone)]
pub struct Series {
    pub id: String,
    pub name: String,
    /// (period, published, value) sorted by period then published
    pub rows: Vec<(Day, Day, f32)>,
}

impl Series {
    /// the latest period published by `as_of`, with its latest vintage; None before the first release
    pub fn at(&self, as_of: Day) -> Option<(Day, f32)> {
        let mut best: Option<(Day, Day, f32)> = None;
        for &(period, published, value) in &self.rows {
            if published > as_of {
                continue;
            }
            let better = match best {
                None => true,
                Some((p, pub_, _)) => period > p || (period == p && published > pub_),
            };
            if better {
                best = Some((period, published, value));
            }
        }
        best.map(|(p, _, v)| (p, v))
    }

    /// the value and its change against the latest period at least a year earlier (as of the same vintage)
    pub fn view(&self, as_of: Day) -> [f32; 2] {
        let Some((period, value)) = self.at(as_of) else {
            return [f32::NAN, f32::NAN];
        };
        let mut earlier: Option<(Day, Day, f32)> = None;
        for &(p, published, v) in &self.rows {
            if published > as_of || p > period - 365 {
                continue;
            }
            let better = match earlier {
                None => true,
                Some((ep, epub, _)) => p > ep || (p == ep && published > epub),
            };
            if better {
                earlier = Some((p, published, v));
            }
        }
        [value, earlier.map_or(f32::NAN, |(_, _, v)| value - v)]
    }
}

#[derive(Clone)]
pub struct Economy {
    /// national series (no industry, no region)
    pub national: Vec<Series>,
    /// industry series keyed by their industry label as published (`08`, `B`, `10-12`, …)
    pub industry: Vec<(String, Series)>,
}

pub struct EconomyView {
    pub national: Vec<[f32; 2]>,
    pub industry: HashMap<String, [f32; 2]>,
}

impl Economy {
    pub fn read(path: &Path) -> Result<Economy, Error> {
        let lines = BufReader::new(File::open(path)?).lines().collect::<Result<Vec<String>, _>>()?;
        Economy::from_lines(lines.iter().map(String::as_str))
    }

    /// the `economy.jsonl` lines, from a file or already in memory
    pub fn from_lines<'a>(lines: impl IntoIterator<Item = &'a str>) -> Result<Economy, Error> {
        let mut by_id: HashMap<String, (EconomyLine, Rows)> = HashMap::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let l: EconomyLine = serde_json::from_str(line)?;
            let (Some(period), Some(published), Some(value)) =
                (parse_ymd(&l.period), parse_ymd(&l.published), l.value)
            else {
                continue;
            };
            let published = published.min(period + publication_lag_days(&l.freq));
            by_id
                .entry(l.series.clone())
                .or_insert_with(|| (l, Vec::new()))
                .1
                .push((period, published, value as f32));
        }
        let mut national = Vec::new();
        let mut industry = Vec::new();
        let mut ids: Vec<_> = by_id.into_iter().collect();
        ids.sort_by(|a, b| a.0.cmp(&b.0));
        for (id, (l, mut rows)) in ids {
            rows.sort_unstable_by_key(|r| (r.0, r.1));
            let s = Series {
                id,
                name: l.name,
                rows,
            };
            match (l.sni, l.region) {
                (None, None) => national.push(s),
                (Some(sni), None) => industry.push((sni, s)),
                _ => {}
            }
        }
        Ok(Economy { national, industry })
    }

    /// the economy from series descriptions and their (period, value, published) rows, by the
    /// rules of `from_lines`: a row without a value is dropped, the published day is capped at the
    /// period plus its frequency's lag, rows sort by period then published, series by key, and a
    /// series is national without code and region, an industry series with a code and no region
    pub fn from_series(series: Vec<SeriesRows>) -> Economy {
        let mut series = series;
        series.sort_by(|a, b| a.key.cmp(&b.key));
        let mut national = Vec::new();
        let mut industry = Vec::new();
        for sr in series {
            let freq = sr.freq.as_deref().unwrap_or("");
            let mut rows: Rows = sr
                .rows
                .iter()
                .filter_map(|&(period, value, published)| {
                    let value = value?;
                    let period = from_naive(period);
                    let published = from_naive(published).min(period + publication_lag_days(freq));
                    Some((period, published, value as f32))
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            rows.sort_unstable_by_key(|r| (r.0, r.1));
            let s = Series { id: sr.key, name: sr.name, rows };
            match (sr.sni, sr.region) {
                (None, None) => national.push(s),
                (Some(sni), None) => industry.push((sni, s)),
                _ => {}
            }
        }
        Economy { national, industry }
    }

    pub fn view(&self, as_of: Day) -> EconomyView {
        EconomyView {
            national: self.national.iter().map(|s| s.view(as_of)).collect(),
            industry: self
                .industry
                .iter()
                .map(|(sni, s)| (sni.clone(), s.view(as_of)))
                .collect(),
        }
    }
}

pub const FIGURES: [&str; 8] = [
    "DebtEquityRatio",
    "EquityRatio",
    "NetMargin",
    "NetSalesPerEmployee",
    "OperatingMargin",
    "QuickRatio",
    "ReturnOnAssets",
    "ReturnOnEquity",
];

#[derive(Deserialize)]
struct SectorLine {
    sni: String,
    year: i32,
    min_employees: Option<i32>,
    max_employees: Option<i32>,
    figure: String,
    q1: Option<f64>,
    median: Option<f64>,
    q3: Option<f64>,
    published: String,
}

/// the published size bands kept: all sizes first, then the disjoint employee bands
pub const BANDS: [(Option<i32>, Option<i32>); 10] = [
    (None, None),
    (Some(0), Some(0)),
    (Some(1), Some(4)),
    (Some(5), Some(9)),
    (Some(10), Some(19)),
    (Some(20), Some(49)),
    (Some(50), Some(99)),
    (Some(100), Some(199)),
    (Some(200), Some(499)),
    (Some(500), None),
];

/// (first quartile, median, third quartile) per `FIGURES` entry
pub type Quartiles = [[f32; 3]; 8];

fn band_index(min: Option<i32>, max: Option<i32>) -> Option<u8> {
    BANDS
        .iter()
        .position(|b| *b == (min, max))
        .and_then(|i| u8::try_from(i).ok())
}

/// the band holding this many employees; unknown → all sizes
pub fn band_for(employees: Option<f32>) -> u8 {
    let Some(e) = employees.filter(|e| e.is_finite()) else {
        return 0;
    };
    let n = e.round() as i32;
    BANDS
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, (lo, hi))| lo.is_none_or(|l| n >= l) && hi.is_none_or(|h| n <= h))
        .map_or(0, |(i, _)| u8::try_from(i).unwrap_or(0))
}

/// quartiles per (industry code, year, size band)
#[derive(Clone)]
pub struct Sector {
    rows: HashMap<(String, i32, u8), Quartiles>,
    /// (year, published) ascending
    years: Vec<(i32, Day)>,
}

pub struct SectorView<'a> {
    sector: &'a Sector,
    year: Option<i32>,
}

impl Sector {
    pub fn read(path: &Path) -> Result<Sector, Error> {
        let lines = BufReader::new(File::open(path)?).lines().collect::<Result<Vec<String>, _>>()?;
        Sector::from_lines(lines.iter().map(String::as_str), None)
    }

    /// the `sector.jsonl` lines; `calendar`: the lines the published years are taken from when the
    /// rows are a part of the table (an input holding only a company's own codes, while the view's
    /// year is the newest published over the whole table); `None` takes them from the rows
    pub fn from_lines<'a>(
        lines: impl IntoIterator<Item = &'a str>,
        calendar: Option<&[&str]>,
    ) -> Result<Sector, Error> {
        let mut rows: HashMap<(String, i32, u8), Quartiles> = HashMap::new();
        let mut years: HashMap<i32, Day> = HashMap::new();
        if let Some(cal) = calendar {
            for line in cal {
                if line.trim().is_empty() {
                    continue;
                }
                let l: SectorLine = serde_json::from_str(line)?;
                if let (Some(_), Some(_), Some(published)) = (
                    FIGURES.iter().position(|f| *f == l.figure),
                    band_index(l.min_employees, l.max_employees),
                    parse_ymd(&l.published),
                ) {
                    years
                        .entry(l.year)
                        .and_modify(|p| *p = (*p).min(published))
                        .or_insert(published);
                }
            }
        }
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let l: SectorLine = serde_json::from_str(line)?;
            let (Some(fi), Some(band), Some(published)) = (
                FIGURES.iter().position(|f| *f == l.figure),
                band_index(l.min_employees, l.max_employees),
                parse_ymd(&l.published),
            ) else {
                continue;
            };
            if calendar.is_none() {
                years
                    .entry(l.year)
                    .and_modify(|p| *p = (*p).min(published))
                    .or_insert(published);
            }
            let q = |v: Option<f64>| v.map_or(f32::NAN, |v| v as f32);
            rows.entry((l.sni, l.year, band))
                .or_insert([[f32::NAN; 3]; 8])[fi] = [q(l.q1), q(l.median), q(l.q3)];
        }
        let mut years: Vec<(i32, Day)> = years.into_iter().collect();
        years.sort_unstable();
        Ok(Sector { rows, years })
    }

    pub fn view(&self, as_of: Day) -> SectorView<'_> {
        SectorView {
            sector: self,
            year: self
                .years
                .iter()
                .filter(|(_, published)| *published <= as_of)
                .map(|(y, _)| *y)
                .max(),
        }
    }
}

impl SectorView<'_> {
    /// the most specific code level with figures (5, 4, 3, then 2 digits), in the company's size
    /// band when published, else all sizes
    pub fn lookup(&self, sni: &str, employees: Option<f32>) -> Option<Quartiles> {
        let year = self.year?;
        let band = band_for(employees);
        for len in [5usize, 4, 3, 2] {
            if sni.len() < len {
                continue;
            }
            for b in [band, 0] {
                if let Some(v) = self.sector.rows.get(&(sni[..len].to_owned(), year, b)) {
                    return Some(*v);
                }
            }
        }
        None
    }
}

/// the section letter (A–V) of a two-digit division, from the model's code table
/// (`risk-tokens/texts/sni_2025.json`)
pub fn section_of(division: u32) -> Option<char> {
    static SECTIONS: std::sync::OnceLock<HashMap<u32, char>> = std::sync::OnceLock::new();
    SECTIONS
        .get_or_init(|| {
            #[derive(serde::Deserialize)]
            struct Row {
                code: String,
                section: String,
            }
            let rows: Vec<Row> = serde_json::from_str(include_str!("../../risk-tokens/texts/sni_2025.json")).unwrap_or_default();
            rows.into_iter()
                .filter(|r| r.code.len() == 2)
                .filter_map(|r| Some((r.code.parse().ok()?, r.section.chars().next()?)))
                .collect()
        })
        .get(&division)
        .copied()
}

pub fn day_of(d: NaiveDate) -> Day {
    from_naive(d)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::type_complexity)]
    use super::*;

    fn d(s: &str) -> Day {
        parse_ymd(s).unwrap()
    }

    #[test]
    fn series_respects_vintages() {
        let s = Series {
            id: "x".into(),
            name: "x".into(),
            rows: vec![
                (d("2024-01-01"), d("2024-02-15"), 1.0),
                (d("2024-01-01"), d("2024-05-15"), 1.5), // revision
                (d("2025-01-01"), d("2025-02-15"), 3.0),
                (d("2025-02-01"), d("2025-03-15"), 4.0),
            ],
        };
        assert_eq!(s.at(d("2024-01-31")), None);
        assert_eq!(s.at(d("2024-03-01")), Some((d("2024-01-01"), 1.0)));
        assert_eq!(s.at(d("2024-06-01")), Some((d("2024-01-01"), 1.5)));
        assert_eq!(s.view(d("2025-03-01")), [3.0, 1.5]);
        assert_eq!(s.view(d("2025-04-01")), [4.0, 2.5]);
    }

    #[test]
    fn sector_view_picks_the_latest_published_year_code_level_and_band() {
        let q = |v: f32| -> Quartiles { [[v - 1.0, v, v + 1.0]; 8] };
        let mut rows = HashMap::new();
        rows.insert(("69".to_owned(), 2023, 0), q(1.0));
        rows.insert(("6910".to_owned(), 2023, 0), q(2.0));
        rows.insert(("6910".to_owned(), 2023, 3), q(5.0));
        rows.insert(("69".to_owned(), 2024, 0), q(3.0));
        let sector = Sector {
            rows,
            years: vec![(2023, d("2024-11-01")), (2024, d("2025-11-01"))],
        };
        assert!(sector.view(d("2024-10-01")).lookup("69101", None).is_none());
        assert_eq!(
            sector.view(d("2025-01-01")).lookup("69101", None),
            Some(q(2.0))
        );
        assert_eq!(
            sector.view(d("2025-01-01")).lookup("69101", Some(7.0)),
            Some(q(5.0)),
            "the 5–9 band of the 4-digit code"
        );
        assert_eq!(
            sector.view(d("2025-01-01")).lookup("69101", Some(70.0)),
            Some(q(2.0)),
            "no 50–99 band published: all sizes"
        );
        assert_eq!(
            sector.view(d("2025-12-01")).lookup("69101", None),
            Some(q(3.0))
        );
        assert_eq!(sector.view(d("2025-12-01")).lookup("70", None), None);
        assert_eq!(band_for(None), 0);
        assert_eq!(band_for(Some(0.0)), 1);
        assert_eq!(band_for(Some(4.4)), 2);
        assert_eq!(band_for(Some(1200.0)), 9);
        assert_eq!(
            band_index(Some(1), Some(19)),
            None,
            "overlapping bands are skipped"
        );
    }

    #[test]
    fn sections() {
        // the 2025 classification: legal and accounting activities moved from M to N, real estate
        // from L to M
        assert_eq!(section_of(69), Some('N'));
        assert_eq!(section_of(68), Some('M'));
        assert_eq!(section_of(4), None);
        assert_eq!(section_of(56), Some('I'));
        assert_eq!(section_of(62), Some('K'));
        assert_eq!(section_of(99), Some('V'));
    }

    #[test]
    fn ranges_cover_runs() {
        let idx = ranges(&[1, 1, 2, 5, 5, 5], |x| *x);
        assert_eq!(idx[&1], (0, 2));
        assert_eq!(idx[&2], (2, 3));
        assert_eq!(idx[&5], (3, 6));
    }

    #[test]
    fn economy_from_series_equals_economy_from_lines() {
        let d = |y, m, day| chrono::NaiveDate::from_ymd_opt(y, m, day).unwrap();
        let lines = [
            r#"{"series":"KPI","source":"scb","name":"KPI","freq":"M","sni":null,"region":null,"period":"2024-01-01","value":100.5,"published":"2024-03-01"}"#,
            r#"{"series":"KPI","source":"scb","name":"KPI","freq":"M","sni":null,"region":null,"period":"2024-02-01","value":null,"published":"2024-03-01"}"#,
            r#"{"series":"IPI08","source":"scb","name":"IPI","freq":"Q","sni":"08","region":null,"period":"2023-10-01","value":90.0,"published":"2025-01-01"}"#,
            r#"{"series":"REG","source":"scb","name":"R","freq":"M","sni":null,"region":"01","period":"2024-01-01","value":1.0,"published":"2024-02-01"}"#,
        ];
        let a = Economy::from_lines(lines.iter().copied()).unwrap();
        let b = Economy::from_series(vec![
            SeriesRows { key: "REG".into(), name: "R".into(), freq: Some("M".into()), sni: None, region: Some("01".into()), rows: vec![(d(2024, 1, 1), Some(1.0), d(2024, 2, 1))] },
            SeriesRows { key: "KPI".into(), name: "KPI".into(), freq: Some("M".into()), sni: None, region: None, rows: vec![(d(2024, 2, 1), None, d(2024, 3, 1)), (d(2024, 1, 1), Some(100.5), d(2024, 3, 1))] },
            SeriesRows { key: "IPI08".into(), name: "IPI".into(), freq: Some("Q".into()), sni: Some("08".into()), region: None, rows: vec![(d(2023, 10, 1), Some(90.0), d(2025, 1, 1))] },
        ]);
        let flat = |e: &Economy| -> Vec<(String, Option<String>, Vec<(Day, Day, f32)>)> {
            e.national.iter().map(|s| (s.id.clone(), None, s.rows.clone()))
                .chain(e.industry.iter().map(|(l, s)| (s.id.clone(), Some(l.clone()), s.rows.clone())))
                .collect()
        };
        assert_eq!(flat(&a), flat(&b));
        assert_eq!(a.national.len(), 1);
        assert_eq!(a.industry.len(), 1);
    }
}
