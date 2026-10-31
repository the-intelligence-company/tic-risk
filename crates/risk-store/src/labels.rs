//! competing-event labels per company: one event record per default-type event with valid_time
//! and known_at, and the life interval that bounds the risk set. the label at an as-of point and
//! horizon is computed on demand (`label_at`), never materialised as an entity-month grid. primary
//! event: bankruptcy initiation. competing: compulsory liquidation decided, reconstruction decided,
//! deregistration by the registry. censoring: voluntary exits and the end of the company for any
//! other reason, at `CompanyLife.ceased`.

use std::collections::BTreeMap;

use chrono::{Days, Months, NaiveDate};

use crate::known_at::KnownAt;
use crate::population::CompanyLife;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DefaultEvent {
    Bankruptcy,
    CompulsoryLiquidation,
    Reconstruction,
    Deregistration,
}

impl DefaultEvent {
    pub const ALL: [DefaultEvent; 4] = [
        DefaultEvent::Bankruptcy,
        DefaultEvent::CompulsoryLiquidation,
        DefaultEvent::Reconstruction,
        DefaultEvent::Deregistration,
    ];

    pub fn from_code(code: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.code() == code)
    }

    pub fn code(self) -> &'static str {
        match self {
            DefaultEvent::Bankruptcy => "bankruptcy",
            DefaultEvent::CompulsoryLiquidation => "compulsory_liquidation",
            DefaultEvent::Reconstruction => "reconstruction",
            DefaultEvent::Deregistration => "deregistration",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventRecord {
    pub company_id: i32,
    pub event: DefaultEvent,
    pub valid_time: NaiveDate,
    pub known_at: KnownAt,
    /// when the state ended (bankruptcy terminated, liquidation ended, …); None = still in it or terminal
    pub ended: Option<NaiveDate>,
}

/// the answer for one company at one as-of point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Label {
    /// the first default event in (as_of, as_of + horizon], if any
    pub event: Option<(DefaultEvent, NaiveDate)>,
    /// the company ended for a non-default reason before the horizon: censored at this date
    pub censored_at: Option<NaiveDate>,
}

pub struct LabelSet {
    lives: BTreeMap<i32, CompanyLife>,
    events: BTreeMap<i32, Vec<EventRecord>>,
}

impl LabelSet {
    /// from label records (`label_events_<form>.csv`); records of unknown companies are dropped
    pub fn from_records(
        lives: Vec<CompanyLife>,
        records: impl IntoIterator<Item = EventRecord>,
    ) -> Self {
        let lives: BTreeMap<i32, CompanyLife> =
            lives.into_iter().map(|l| (l.company_id, l)).collect();
        let mut events: BTreeMap<i32, Vec<EventRecord>> = BTreeMap::new();
        for rec in records {
            if lives.contains_key(&rec.company_id) {
                events.entry(rec.company_id).or_default().push(rec);
            }
        }
        for v in events.values_mut() {
            v.sort_by_key(|e| (e.valid_time, e.event));
            v.dedup();
        }
        LabelSet { lives, events }
    }

    pub fn lives(&self) -> impl Iterator<Item = &CompanyLife> {
        self.lives.values()
    }

    pub fn life(&self, company_id: i32) -> Option<&CompanyLife> {
        self.lives.get(&company_id)
    }

    /// the default events of one company, in valid-time order
    pub fn events_of(&self, company_id: i32) -> &[EventRecord] {
        self.events.get(&company_id).map_or(&[], Vec::as_slice)
    }

    pub fn events(&self) -> impl Iterator<Item = &EventRecord> {
        self.events.values().flatten()
    }

    /// in the risk set at `as_of`: alive, and not inside a default state (an open bankruptcy or
    /// compulsory liquidation, a running reconstruction) on that day
    pub fn in_risk_set(&self, company_id: i32, as_of: NaiveDate) -> bool {
        let Some(life) = self.lives.get(&company_id) else {
            return false;
        };
        if !life.alive_on(as_of) {
            return false;
        }
        self.events.get(&company_id).is_none_or(|evs| {
            !evs.iter()
                .any(|e| e.valid_time <= as_of && e.ended.is_none_or(|end| end > as_of))
        })
    }

    /// None when not in the risk set at `as_of`
    pub fn label_at(
        &self,
        company_id: i32,
        as_of: NaiveDate,
        horizon_months: u32,
    ) -> Option<Label> {
        if !self.in_risk_set(company_id, as_of) {
            return None;
        }
        let horizon_end = as_of.checked_add_months(Months::new(horizon_months))?;
        let first_event = self
            .events
            .get(&company_id)
            .and_then(|evs| {
                evs.iter()
                    .find(|e| e.valid_time > as_of && e.valid_time <= horizon_end)
            })
            .map(|e| (e.event, e.valid_time));
        let life = self.lives.get(&company_id)?;
        let censored_at = match (first_event, life.ceased) {
            (None, Some(c)) if c > as_of && c <= horizon_end => Some(c),
            _ => None,
        };
        Some(Label {
            event: first_event,
            censored_at,
        })
    }

    /// monthly as-of points from `from` to `to` (inclusive, first of month) for one company
    pub fn monthly_points(from: NaiveDate, to: NaiveDate) -> impl Iterator<Item = NaiveDate> {
        let mut cur = from.with_day_first();
        std::iter::from_fn(move || {
            if cur > to {
                return None;
            }
            let p = cur;
            cur = cur.checked_add_months(Months::new(1))?;
            Some(p)
        })
    }
}

trait FirstOfMonth {
    fn with_day_first(self) -> NaiveDate;
}

impl FirstOfMonth for NaiveDate {
    fn with_day_first(self) -> NaiveDate {
        use chrono::Datelike;
        self.checked_sub_days(Days::new(u64::from(self.day0())))
            .unwrap_or(self)
    }
}

