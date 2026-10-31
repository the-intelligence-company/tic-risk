//! what moves an answer: the answer again with one
//! group of the company's own facts left out of the input at a time, or its relations, and the view
//! rebuilt from what remains. the state the model derives from those facts (open cases, the filing
//! state, the profile's industry) goes with them, so a group's effect is its facts' whole weight;
//! one forward pass per group. ranked by raw log odds: the calibration is a step function, so the
//! calibrated difference is often zero for a small effect

use chrono::NaiveDate;
use risk_eval::store::{Economy, Sector};
use risk_tokens::vocab::Vocab;
use serde::Serialize;
use serde_json::Value;

use crate::answer::{Answer, Asked, answer};
use crate::error::Result;
use crate::input::Input;
use crate::live::Live;
use crate::forward::Forward;

/// a group of the company's own facts by event family (the part of the kind before the first
/// dot), or its relations
pub struct Group {
    pub key: &'static str,
    pub label: &'static str,
    families: &'static [&'static str],
    relations: bool,
}

const fn g(key: &'static str, label: &'static str, families: &'static [&'static str]) -> Group {
    Group { key, label, families, relations: false }
}

pub const GROUPS: [Group; 13] = [
    g("annual_reports", "Annual reports", &["dividend"]),
    g("filing", "Filing and late fees", &["fee"]),
    g("enforcement", "Enforcement debt", &["debt", "enforcement"]),
    g("payment_orders", "Payment orders", &["payment_order"]),
    g("tax_account", "Tax account", &[]),
    g("tax_registration", "Tax registration", &[]),
    g("payroll", "Payroll", &["payroll"]),
    g("board", "Board and roles", &["role", "board"]),
    g("ownership", "Ownership", &["ownership", "group"]),
    g("registry", "Registry cases", &["case", "name", "purpose", "articles", "address", "permit", "status"]),
    g("insolvency", "Insolvency", &["bankruptcy", "liquidation", "reconstruction", "liquidation_order"]),
    g("industry", "Industry", &["industry"]),
    Group { key: "relations", label: "Related companies", families: &[], relations: true },
];

impl Group {
    /// the report family splits by kind: the figures are the annual reports, the rest the filing;
    /// the tax family splits into the account and the registration
    fn has(&self, kind: &str) -> bool {
        let family = kind.split('.').next().unwrap_or(kind);
        match (self.key, family) {
            ("annual_reports", "report") => kind == "report.figures",
            ("filing", "report") => kind != "report.figures",
            ("tax_registration", "tax") => kind == "tax.registration",
            ("tax_account", "tax") => kind != "tax.registration",
            _ => self.families.contains(&family),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Effect {
    pub group: &'static str,
    pub label: &'static str,
    /// the group's lines known by the as-of day
    pub facts: usize,
    /// the answer without the group
    pub p_raw: f64,
    pub p_pit: Option<f64>,
    /// the full answer's raw log odds less the occluded one's: above zero, the facts raise the risk
    pub log_odds: f64,
}

/// one line's company, kind and days, read once
struct Line {
    company: Option<i64>,
    kind: String,
    known_by: Option<String>,
}

fn day_text(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str).map(str::to_owned)
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

/// each group's effect on `base` (the full answer at `day`), largest first; a group with no fact
/// known by the day is left out, and so is one whose occluded sample cannot be scored
#[allow(clippy::too_many_arguments)]
pub fn explain(engine: &dyn Forward, live: &Live, input: &Input, corpus: &Vocab, context: &(Economy, Sector), day: NaiveDate, a: &Asked, base: &Answer) -> Result<Vec<Effect>> {
    let focal = i64::from(input.meta.focal);
    let iso = day.to_string();
    // a line counts by the later of its two days: known (`ka`) and happened (`t`, `from`)
    let known = |v: &Value, happened: &str| {
        let ka = day_text(v.get("ka"));
        let t = day_text(v.get(happened));
        match (ka, t) {
            (Some(k), Some(t)) => Some(if k > t { k } else { t }),
            (k, t) => k.or(t),
        }
    };
    let events: Vec<Line> = input
        .events
        .iter()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap_or(Value::Null);
            Line { company: v.get("c").and_then(Value::as_i64), kind: v.get("k").and_then(Value::as_str).unwrap_or("").to_owned(), known_by: known(&v, "t") }
        })
        .collect();
    let relations_known = input
        .relations
        .iter()
        .filter(|l| serde_json::from_str::<Value>(l).ok().and_then(|v| known(&v, "from")).is_some_and(|k| k <= iso))
        .count();
    let base_logit = logit(base.p_raw);
    let mut out = Vec::new();
    for group in &GROUPS {
        let (keep_events, keep_relations, facts) = if group.relations {
            (input.events.clone(), Vec::new(), relations_known)
        } else {
            let mine = |e: &Line| e.company == Some(focal) && group.has(&e.kind);
            let facts = events.iter().filter(|e| mine(e) && e.known_by.as_deref().is_some_and(|k| k <= iso.as_str())).count();
            let kept = input.events.iter().zip(&events).filter(|(_, e)| !mine(e)).map(|(l, _)| l.clone()).collect();
            (kept, input.relations.clone(), facts)
        };
        if facts == 0 {
            continue;
        }
        let occluded = Input { meta: input.meta.clone(), lives: input.lives.clone(), events: keep_events, relations: keep_relations, economy: Vec::new(), sector: Vec::new(), calendar: Vec::new() };
        let (fetched, _) = occluded.build(corpus, &live.md, context.0.clone(), context.1.clone())?;
        let Ok((without, _)) = answer(engine, live, &fetched, day, a, true) else { continue };
        out.push(Effect { group: group.key, label: group.label, facts, p_raw: without.p_raw, p_pit: without.p_pit, log_odds: base_logit - logit(without.p_raw) });
    }
    out.sort_by(|x, y| y.log_odds.abs().total_cmp(&x.log_odds.abs()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_family_has_one_group() {
        let families = [
            "bankruptcy.initiated", "liquidation.decided", "reconstruction.decided", "status.struck_off", "report.figures", "report.finding",
            "fee.registered", "dividend.registered", "liquidation_order.issued", "case.received", "board.composition", "name.changed",
            "purpose.changed", "articles.changed", "debt.registered", "role.started", "ownership.owner_added", "payment_order.applied",
            "enforcement.case_registered", "payroll.declared", "tax.registration", "tax.surcharge", "industry.code", "permit.serving_granted",
            "group.parent", "address.registered",
        ];
        for kind in families {
            let n = GROUPS.iter().filter(|g| !g.relations && g.has(kind)).count();
            assert_eq!(n, 1, "{kind} is in {n} groups");
        }
    }
}
