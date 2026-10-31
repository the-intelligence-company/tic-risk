//! `tic-risk check`: what the model makes of an input someone made.
//! every line read on its own (a line that does not parse is reported with its number, and the
//! rest still checked), then the same build as `score`, whose own account says what it left out;
//! and per as-of day the sample's size. no GPU, no weights

use std::collections::BTreeMap;

use chrono::NaiveDate;
use risk_engine::input::Input;
use risk_engine::live::Live;
use risk_tokens::vocab::Vocab;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// the input cannot be scored as it is
    Error,
    /// the model leaves something out, or reads something that is likely wrong
    Warning,
    /// worth knowing, nothing lost
    Note,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub level: Level,
    pub what: String,
    pub count: usize,
    /// a few of the lines or keys behind it
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaySample {
    pub as_of: NaiveDate,
    pub tokens: Option<usize>,
    pub neighbours: Option<usize>,
    pub truncated: Option<bool>,
    pub unknown_ids: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub input: String,
    pub company: i32,
    pub events: usize,
    pub relations: usize,
    pub companies: usize,
    pub findings: Vec<Finding>,
    pub days: Vec<DaySample>,
}

impl Check {
    pub fn ok(&self) -> bool {
        !self.findings.iter().any(|f| f.level == Level::Error)
    }
}

const EXAMPLES: usize = 3;

/// counts and a few examples per finding, keyed by its text
#[derive(Default)]
struct Tally(BTreeMap<(Level, String), (usize, Vec<String>)>);

impl Tally {
    fn add(&mut self, level: Level, what: impl Into<String>, example: impl Into<String>) {
        let e = self.0.entry((level, what.into())).or_default();
        e.0 += 1;
        if e.1.len() < EXAMPLES {
            e.1.push(example.into());
        }
    }

    fn add_n(&mut self, level: Level, what: impl Into<String>, n: usize, examples: Vec<String>) {
        if n > 0 {
            self.0.insert((level, what.into()), (n, examples.into_iter().take(EXAMPLES).collect()));
        }
    }

    fn findings(self) -> Vec<Finding> {
        self.0.into_iter().map(|((level, what), (count, examples))| Finding { level, what, count, examples }).collect()
    }
}

fn date(v: Option<&Value>) -> Result<Option<NaiveDate>, ()> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => NaiveDate::parse_from_str(s, "%Y-%m-%d").map(Some).map_err(|_| ()),
        Some(_) => Err(()),
    }
}

/// the lines that parse, with what is wrong with the others and with the dates of every line
fn lines(file: &str, lines: &[String], happened: &str, read_at: Option<NaiveDate>, t: &mut Tally) -> Vec<String> {
    let mut good = Vec::with_capacity(lines.len());
    for (i, l) in lines.iter().enumerate() {
        let at = format!("{file} line {}", i + 1);
        let v: Value = match serde_json::from_str(l) {
            Ok(v) => v,
            Err(e) => {
                t.add(Level::Error, format!("{file}: invalid JSON (line skipped)"), format!("{at}: {e}"));
                continue;
            }
        };
        let (Ok(when), Ok(known)) = (date(v.get(happened)), date(v.get("ka"))) else {
            t.add(Level::Error, format!("{file}: invalid date, expected YYYY-MM-DD (line skipped)"), at);
            continue;
        };
        if known.is_none() {
            t.add(Level::Error, format!("{file}: missing known date \"ka\" (line skipped)"), at);
            continue;
        }
        if let (Some(w), Some(k)) = (when, known)
            && k < w
        {
            t.add(Level::Warning, format!("{file}: known date \"ka\" is before the event date \"{happened}\""), at.clone());
        }
        if let Some(r) = read_at
            && (when.is_some_and(|w| w > r) || known.is_some_and(|k| k > r))
        {
            t.add(Level::Warning, format!("{file}: date after read_at in meta.json"), at.clone());
        }
        good.push(l.clone());
    }
    good
}

pub fn run(input: &Input, input_name: &str, corpus: &Vocab, live: &Live, days: &[NaiveDate]) -> anyhow::Result<Check> {
    let mut t = Tally::default();
    let focal = input.meta.focal;
    let read_at = chrono::DateTime::parse_from_rfc3339(&input.meta.read_at).ok().map(|d| d.date_naive());
    if read_at.is_none() {
        t.add(Level::Note, "meta.json: read_at is not a valid RFC 3339 timestamp (later dates are not checked)", input.meta.read_at.clone());
    }
    match input.lives.iter().find(|l| l.company_id == focal) {
        None => t.add(Level::Error, "lives.csv: no row for the company in meta.json (focal)", focal.to_string()),
        Some(l) => {
            if l.legal_form.code() != "AB" {
                t.add(Level::Warning, "not a limited company (AB); the model was trained on limited companies only", l.legal_form.code());
            }
            if l.registered.is_none() {
                t.add(Level::Warning, "lives.csv: missing registration date", focal.to_string());
            }
        }
    }
    let events = lines("events.jsonl", &input.events, "t", read_at, &mut t);
    let relations = lines("relations.jsonl", &input.relations, "from", read_at, &mut t);
    let companies: std::collections::BTreeSet<i32> = input.lives.iter().map(|l| l.company_id).collect();
    let mut strangers = 0usize;
    let mut stranger_examples = Vec::new();
    for l in &events {
        let c = serde_json::from_str::<Value>(l).ok().and_then(|v| v.get("c").and_then(Value::as_i64));
        if let Some(c) = c
            && !i32::try_from(c).is_ok_and(|c| companies.contains(&c))
        {
            strangers += 1;
            if stranger_examples.len() < EXAMPLES {
                stranger_examples.push(c.to_string());
            }
        }
    }
    t.add_n(Level::Warning, "events.jsonl: events for companies not in lives.csv", strangers, stranger_examples);

    // the economy series the model was trained with: the others are left out
    let mut series: BTreeMap<String, bool> = BTreeMap::new();
    for l in &input.economy {
        if let Some(k) = serde_json::from_str::<Value>(l).ok().and_then(|v| v.get("series").and_then(Value::as_str).map(str::to_owned)) {
            let known = corpus.field(&format!("series.{k}")).is_some();
            series.entry(k).or_insert(known);
        }
    }
    let unknown_series: Vec<String> = series.iter().filter(|(_, k)| !**k).map(|(s, _)| s.clone()).collect();
    t.add_n(Level::Warning, "economy.jsonl: unknown series (skipped)", unknown_series.len(), unknown_series);
    if series.is_empty() {
        t.add(Level::Warning, "economy.jsonl: no economic series (the model expects them)", "economy.jsonl");
    }

    // the build's own account
    let clean = Input { meta: input.meta.clone(), lives: input.lives.clone(), events, relations, economy: input.economy.clone(), sector: input.sector.clone(), calendar: input.calendar.clone() };
    let (economy, sector) = clean.context()?;
    let (fetched, stats) = clean.build(corpus, &live.md, economy, sector)?;
    let keyed = |m: &BTreeMap<String, u64>| (m.values().sum::<u64>() as usize, m.iter().map(|(k, n)| format!("{k} ×{n}")).collect::<Vec<_>>());
    let (n, ex) = keyed(&stats.unknown_kinds);
    t.add_n(Level::Warning, "unknown event types (skipped)", n, ex);
    let (n, ex) = keyed(&stats.skipped_attrs);
    t.add_n(Level::Note, "attributes the model doesn't use (ignored)", n, ex);
    let (n, ex) = keyed(&stats.implausible_values);
    t.add_n(Level::Warning, "implausible values (skipped)", n, ex);
    t.add_n(Level::Note, "follow-up dates earlier than the event itself (treated as plain dates)", stats.past_future_dates as usize, Vec::new());
    let new_codes: Vec<String> = fetched.streams.vocab.entries.iter().skip(corpus.len()).map(|e| e.key.clone()).collect();
    t.add_n(Level::Warning, "unknown codes (treated as unknown)", new_codes.len(), new_codes);

    let q = Live::question("bankruptcy_12m")?;
    let samples = days
        .iter()
        .map(|&d| match live.pack(&fetched, d, q) {
            Ok(p) => DaySample { as_of: d, tokens: Some(p.n_tok), neighbours: Some(p.neighbours), truncated: Some(p.truncated), unknown_ids: Some(p.unknown_ids), error: None },
            Err(e) => {
                t.add(Level::Error, "dates the company can't be scored on", format!("{d}: {e}"));
                DaySample { as_of: d, tokens: None, neighbours: None, truncated: None, unknown_ids: None, error: Some(e.to_string()) }
            }
        })
        .collect();
    let mut findings = t.findings();
    findings.sort_by_key(|f| f.level);
    Ok(Check { input: input_name.to_owned(), company: focal, events: input.events.len(), relations: input.relations.len(), companies: input.lives.len(), findings, days: samples })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn bad_lines_are_reported_and_left_out_the_rest_kept() {
        let ls = [
            r#"{"c":1,"k":"payroll.declared","t":"2025-01-31","ka":"2025-02-12","est":false,"s":"skatteverket","a":{}}"#,
            "not json",
            r#"{"c":1,"k":"payroll.declared","t":"2025-13-31","ka":"2025-02-12","est":false,"s":"skatteverket","a":{}}"#,
            r#"{"c":1,"k":"debt.registered","t":"2025-03-01","ka":"2025-02-01","est":false,"s":"kronofogden","a":{}}"#,
            r#"{"c":1,"k":"debt.registered","t":"2027-03-01","ka":"2027-03-02","est":false,"s":"kronofogden","a":{}}"#,
            r#"{"c":1,"k":"debt.registered","t":"2025-03-01","est":false,"s":"kronofogden","a":{}}"#,
        ]
        .map(str::to_owned);
        let mut t = Tally::default();
        let good = lines("events.jsonl", &ls, "t", NaiveDate::from_ymd_opt(2026, 10, 1), &mut t);
        assert_eq!(good.len(), 3, "the valid line, the one known early and the one dated late are kept");
        let f = t.findings();
        let count = |level: Level, starts: &str| f.iter().filter(|x| x.level == level && x.what.contains(starts)).map(|x| x.count).sum::<usize>();
        assert_eq!(count(Level::Error, "invalid JSON"), 1);
        assert_eq!(count(Level::Error, "YYYY-MM-DD"), 1);
        assert_eq!(count(Level::Error, "\"ka\""), 1);
        assert_eq!(count(Level::Warning, "before the event date"), 1);
        assert_eq!(count(Level::Warning, "after read_at"), 1);
        assert_eq!(f.iter().find(|x| x.what.contains("invalid JSON")).unwrap().examples[0].split(':').next(), Some("events.jsonl line 2"));
    }
}
