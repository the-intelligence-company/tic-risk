//! declarative credit questions: entity, target, horizon, as-of rule. one function turns a
//! question into the outcome of a (company, as-of) row; nothing else writes a label. the risk set
//! and the first event come from the label set; what the question adds is which events count as
//! the target.

use chrono::NaiveDate;
use risk_store::labels::{DefaultEvent, LabelSet};
use serde::Serialize;

use crate::cache::{Dict, EventRec};
use crate::dates::{add_months, from_naive};
use crate::store::Store;

/// an early-warning event: the first occurrence inside the horizon,
/// unless a default-type event or the company's end came first
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EventTarget {
    /// `debt.registered`, or `debt.changed` with the amount above the previous amount
    EnforcementDebt,
    /// the `tax.registration` record dated at the day a tax registration ended
    /// (it carries the ended state, the end date and the recall reasons), with a misconduct
    /// reason: 514 income return not filed, 515/524 deficient reporting or payment, 516 trading
    /// prohibition, 517 abuse, 518/519 neglect, 526 no reply to an injunction, 527 a household-services
    /// deduction not repaid, 528 an earlier ground remains. the company's own request (511, 525) and the
    /// administrative deregistrations (504, 508, 512) are not the event
    FskattRevoked,
    /// `fee.registered`: the registry's late-filing fee, the registered form of "report overdue"
    ReportOverdue,
}

impl EventTarget {
    pub fn code(self) -> &'static str {
        match self {
            EventTarget::EnforcementDebt => "enforcement_debt",
            EventTarget::FskattRevoked => "fskatt_revoked",
            EventTarget::ReportOverdue => "report_overdue",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Target {
    /// bankruptcy initiated (registry rows)
    Bankruptcy,
    /// bankruptcy, compulsory liquidation, reconstruction or deregistration: the first of them
    AnyDefault,
    /// an early-warning event from the event cache
    Event(EventTarget),
}

impl Target {
    pub fn matches(self, e: DefaultEvent) -> bool {
        match self {
            Target::Bankruptcy => e == DefaultEvent::Bankruptcy,
            Target::AnyDefault => true,
            Target::Event(_) => false,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Target::Bankruptcy => "bankruptcy",
            Target::AnyDefault => "any_default",
            Target::Event(e) => e.code(),
        }
    }

    pub fn is_event(self) -> bool {
        matches!(self, Target::Event(_))
    }
}

/// the tax registration's recall reasons that mark misconduct (503, because of bankruptcy, leaks
/// the outcome and never reaches the records; 504/508/511/512/525 are administrative or voluntary)
pub const FSKATT_MISCONDUCT: &[i32] = &[514, 515, 516, 517, 518, 519, 524, 526, 527, 528];

/// an event target resolved against a cache dictionary: kind ids and the categorical filter
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRule {
    /// kinds that count by themselves
    pub first: Vec<u16>,
    /// the kind whose second numeric (amount minus previous amount) must be positive
    pub rising: Option<u16>,
    /// the categorical code the `first` kinds must carry (registration none)
    pub cat: Option<u16>,
    /// the numeric (a code) the `first` kinds must carry one of
    pub num_in: Option<&'static [i32]>,
}

impl EventRule {
    pub fn resolve(target: EventTarget, dict: &Dict) -> EventRule {
        let kind = |n: &str| dict.event_kind(n);
        match target {
            EventTarget::EnforcementDebt => EventRule {
                first: kind("debt.registered").into_iter().collect(),
                rising: kind("debt.changed"),
                cat: None,
                num_in: None,
            },
            EventTarget::FskattRevoked => EventRule {
                first: kind("tax.registration").into_iter().collect(),
                rising: None,
                cat: None,
                num_in: Some(FSKATT_MISCONDUCT),
            },
            EventTarget::ReportOverdue => EventRule {
                first: kind("fee.registered").into_iter().collect(),
                rising: None,
                cat: None,
                num_in: None,
            },
        }
    }

    pub fn matches(&self, e: &EventRec) -> bool {
        if self.first.contains(&e.kind) {
            return self.cat.is_none_or(|c| e.cat == c)
                && self
                    .num_in
                    .is_none_or(|set| set.contains(&(e.num.round() as i32)));
        }
        self.rising == Some(e.kind) && e.num2 > 0.0
    }
}

/// the event cache behind an event question
pub struct Events<'a> {
    pub store: &'a Store,
    pub rule: EventRule,
}

impl<'a> Events<'a> {
    pub fn new(store: &'a Store, target: EventTarget) -> Events<'a> {
        Events {
            store,
            rule: EventRule::resolve(target, &store.dict),
        }
    }

    /// for a question: `Some` when it needs the cache
    pub fn for_question(store: &'a Store, q: &Question) -> Option<Events<'a>> {
        match q.target {
            Target::Event(t) => Some(Events::new(store, t)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Question {
    pub name: &'static str,
    pub target: Target,
    pub horizon_months: u32,
    pub description: &'static str,
}

/// the questions; the as-of rule is monthly points on the first of the month, the
/// entity is every legal entity in the risk set on that day
pub const QUESTIONS: &[Question] = &[
    Question {
        name: "bankruptcy_3m",
        target: Target::Bankruptcy,
        horizon_months: 3,
        description: "bankruptcy initiated within 3 months",
    },
    Question {
        name: "bankruptcy_6m",
        target: Target::Bankruptcy,
        horizon_months: 6,
        description: "bankruptcy initiated within 6 months",
    },
    Question {
        name: "bankruptcy_12m",
        target: Target::Bankruptcy,
        horizon_months: 12,
        description: "bankruptcy initiated within 12 months (the headline PD)",
    },
    Question {
        name: "bankruptcy_24m",
        target: Target::Bankruptcy,
        horizon_months: 24,
        description: "bankruptcy initiated within 24 months",
    },
    Question {
        name: "default_12m",
        target: Target::AnyDefault,
        horizon_months: 12,
        description: "any default-type event within 12 months",
    },
    Question {
        name: "enforcement_debt_12m",
        target: Target::Event(EventTarget::EnforcementDebt),
        horizon_months: 12,
        description: "enforcement debt arising or rising within 12 months (early warning)",
    },
    Question {
        name: "fskatt_revoked_12m",
        target: Target::Event(EventTarget::FskattRevoked),
        horizon_months: 12,
        description: "tax registration revoked for misconduct within 12 months (early warning)",
    },
    Question {
        name: "report_overdue_12m",
        target: Target::Event(EventTarget::ReportOverdue),
        horizon_months: 12,
        description: "late-filing fee for the annual report within 12 months (early warning)",
    },
];

pub fn find(name: &str) -> Option<&'static Question> {
    QUESTIONS.iter().find(|q| q.name == name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub positive: bool,
    /// the first default-type event inside the horizon, target or not
    pub event: Option<DefaultEvent>,
    pub event_date: Option<NaiveDate>,
    /// a non-target default event came first: the target could no longer happen
    pub competing: bool,
    /// the company ended for a non-default reason inside the horizon
    pub censored: bool,
}

/// None when the company is not in the risk set at `as_of`; an event question needs `events`
/// (None there answers nothing)
pub fn outcome(
    labels: &LabelSet,
    events: Option<&Events>,
    company: i32,
    as_of: NaiveDate,
    q: &Question,
) -> Option<Outcome> {
    let label = labels.label_at(company, as_of, q.horizon_months)?;
    if let Target::Event(_) = q.target {
        let ev = events?;
        let start = from_naive(as_of);
        let end = from_naive(add_months(as_of, q.horizon_months));
        // the target counts only before a default-type event or the company's end
        let limit = [
            label.event.map(|(_, d)| from_naive(d)),
            label.censored_at.map(from_naive),
        ]
        .into_iter()
        .flatten()
        .min();
        let first = ev
            .store
            .events_of(company)
            .iter()
            .filter(|e| e.valid > start && e.valid <= end && ev.rule.matches(e))
            .map(|e| e.valid)
            .min();
        let positive = first.is_some_and(|t| limit.is_none_or(|l| t <= l));
        return Some(Outcome {
            positive,
            event: label.event.map(|(e, _)| e),
            event_date: label.event.map(|(_, d)| d),
            competing: !positive && label.event.is_some(),
            censored: !positive && label.censored_at.is_some(),
        });
    }
    let positive = label.event.is_some_and(|(e, _)| q.target.matches(e));
    Some(Outcome {
        positive,
        event: label.event.map(|(e, _)| e),
        event_date: label.event.map(|(_, d)| d),
        competing: label.event.is_some() && !positive,
        censored: label.censored_at.is_some(),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn event_rules_match_kinds_rises_and_codes() {
        let rec = |kind, num2, cat| EventRec {
            company: 1,
            kind,
            est: false,
            valid: 0,
            known: 0,
            num: 0.0,
            num2,
            cat,
            cat2: 0,
            extra: u32::MAX,
        };
        let debt = EventRule {
            first: vec![3],
            rising: Some(4),
            cat: None,
            num_in: None,
        };
        assert!(debt.matches(&rec(3, 0.0, 0)));
        assert!(debt.matches(&rec(4, 100.0, 0)));
        assert!(!debt.matches(&rec(4, -5.0, 0)));
        assert!(!debt.matches(&rec(5, 100.0, 0)));
        let fskatt = EventRule {
            first: vec![9],
            rising: None,
            cat: None,
            num_in: Some(FSKATT_MISCONDUCT),
        };
        let with_reason = |reason: f32, cat| EventRec {
            num: reason,
            ..rec(9, 0.0, cat)
        };
        assert!(fskatt.matches(&with_reason(515.0, 7)));
        assert!(fskatt.matches(&with_reason(514.0, 2)));
        assert!(
            !fskatt.matches(&with_reason(511.0, 7)),
            "own request is not misconduct"
        );
        assert!(
            !fskatt.matches(&with_reason(508.0, 7)),
            "automatic deregistration is not misconduct"
        );
        assert!(!fskatt.matches(&rec(9, 0.0, 7)), "no reason, no event");
        assert_eq!(
            Target::Event(EventTarget::ReportOverdue).code(),
            "report_overdue"
        );
        assert!(find("fskatt_revoked_12m").is_some_and(|q| q.target.is_event()));
    }
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
                rec(2, DefaultEvent::Bankruptcy, d(2025, 9, 1)),
            ],
        )
    }

    #[test]
    fn target_and_competing_events() {
        let s = set();
        let q = find("bankruptcy_12m").unwrap();
        let o = outcome(&s, None, 1, d(2025, 1, 1), q).unwrap();
        assert!(o.positive && !o.competing && !o.censored);
        let o = outcome(&s, None, 2, d(2025, 1, 1), q).unwrap();
        assert!(!o.positive && o.competing);
        assert_eq!(o.event, Some(DefaultEvent::CompulsoryLiquidation));
        let o = outcome(&s, None, 2, d(2025, 1, 1), find("default_12m").unwrap()).unwrap();
        assert!(o.positive);
        let o = outcome(&s, None, 3, d(2025, 1, 1), q).unwrap();
        assert!(!o.positive && o.censored);
        assert!(
            outcome(&s, None, 3, d(2025, 4, 1), q).is_none(),
            "ceased: not at risk"
        );
        assert!(
            outcome(&s, None, 1, d(2025, 7, 1), q).is_none(),
            "in bankruptcy: not at risk"
        );
        assert!(
            !outcome(&s, None, 1, d(2024, 5, 1), find("bankruptcy_3m").unwrap())
                .unwrap()
                .positive
        );
    }
}
