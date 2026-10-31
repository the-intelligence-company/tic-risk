//! a compact cache of the record files: `events.jsonl` and `relations.jsonl` become fixed-size little-endian records sorted by company, plus a
//! dictionary of kinds and categorical codes. one numeric, one secondary numeric and two
//! categorical attributes per event are kept, chosen per kind by the table below; the rest of an
//! event's attributes are dropped. the cache is disposable, rebuilt from the record files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::dates::{DAY_MAX, DAY_MIN, Day, parse_ymd};

pub const EVENT_MAGIC: &[u8; 8] = b"TICEVT02";
pub const FIGURES_MAGIC: &[u8; 8] = b"TICFIG01";
pub const REL_MAGIC: &[u8; 8] = b"TICREL01";
pub const EVENT_REC_SIZE: usize = 32;
/// `EventRec::extra` of an event without a figures row
pub const NO_EXTRA: u32 = u32::MAX;
pub const REL_REC_SIZE: usize = 40;
const CHUNK_BYTES: usize = 64 << 20;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EventRec {
    pub company: i32,
    pub kind: u16,
    pub est: bool,
    pub valid: Day,
    pub known: Day,
    pub num: f32,
    pub num2: f32,
    pub cat: u16,
    pub cat2: u16,
    /// `report.figures`: the row in `figures.bin`; else `NO_EXTRA`
    pub extra: u32,
}

/// the attributes of a `report.figures` event kept as a row of `figures.bin`, in this order
pub fn figure_attrs() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = risk_store::figures::FIGURES
        .iter()
        .map(|f| f.name)
        .collect();
    v.extend(["net_sales_change", "operating_result_change", "months"]);
    v
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum EndpointType {
    #[default]
    Company = 0,
    Person = 1,
    Address = 2,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RelRec {
    pub company: i32,
    pub kind: u16,
    pub b_type: EndpointType,
    pub est: bool,
    pub b_id: i64,
    /// `DAY_MIN` when the start was never given
    pub from: Day,
    /// `DAY_MAX` while open
    pub to: Day,
    pub known: Day,
    pub num: f32,
    pub cat: u16,
}

impl RelRec {
    pub fn active_on(&self, day: Day) -> bool {
        self.from <= day && day < self.to
    }
}

/// which attribute feeds which cache field
#[derive(Debug, Clone, Copy)]
pub enum Spec {
    None,
    /// the attribute's number (bools as 1/0), or its `key=value` as a category
    Attr(&'static str),
    /// numeric only: valid time minus the date attribute, in days
    LagDays(&'static str),
    /// numeric only: first minus second
    Diff(&'static str, &'static str),
    /// categorical only: both values joined, `k1=v1,k2=v2`
    Pair(&'static str, &'static str),
    /// numeric only: the first code of a comma-separated code list (`"514,515"` → 514)
    FirstCode(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct AttrPlan {
    pub num: Spec,
    pub num2: Spec,
    pub cat: Spec,
    pub cat2: Spec,
}

const fn plan(num: Spec, num2: Spec, cat: Spec, cat2: Spec) -> AttrPlan {
    AttrPlan {
        num,
        num2,
        cat,
        cat2,
    }
}

use Spec::{Attr, Diff, FirstCode, LagDays, None as No, Pair};

/// the closed kind table; ids are positions. kinds met in the file but missing here get an id
/// after these and no attributes
pub const EVENT_KINDS: &[(&str, AttrPlan)] = &[
    ("bankruptcy.initiated", plan(No, No, No, No)),
    ("bankruptcy.terminated", plan(No, No, Attr("surplus"), No)),
    ("bankruptcy.repealed", plan(No, No, No, No)),
    (
        "liquidation.decided",
        plan(No, No, Attr("decider"), Attr("compulsory")),
    ),
    ("liquidation.ended", plan(No, No, No, No)),
    ("reconstruction.decided", plan(No, No, No, No)),
    ("reconstruction.ended", plan(No, No, No, No)),
    ("status.struck_off", plan(No, No, Attr("status_type"), No)),
    (
        "report.received",
        plan(LagDays("period_end"), No, Attr("interim"), No),
    ),
    (
        "report.approved",
        plan(
            LagDays("period_end"),
            No,
            Attr("interim"),
            Attr("consolidated"),
        ),
    ),
    (
        "report.registered",
        plan(LagDays("period_end"), No, Attr("ruleset"), Attr("interim")),
    ),
    ("report.shortage", plan(No, No, Attr("code"), No)),
    ("report.case_terminated", plan(No, No, Attr("reason"), No)),
    ("fee.registered", plan(Attr("number"), No, No, No)),
    ("fee.voided", plan(Attr("number"), No, Attr("cause"), No)),
    (
        "dividend.registered",
        plan(Attr("amount"), No, Attr("type"), No),
    ),
    ("liquidation_order.issued", plan(No, No, Attr("cause"), No)),
    (
        "liquidation_order.recalled",
        plan(No, No, Attr("cause"), No),
    ),
    ("case.received", plan(No, No, Attr("codes"), No)),
    (
        "case.terminated",
        plan(No, No, Attr("outcome"), Attr("codes")),
    ),
    ("board.gap_started", plan(No, No, Attr("gap"), No)),
    (
        "board.gap_ended",
        plan(LagDays("started"), No, Attr("gap"), No),
    ),
    (
        "board.composition",
        plan(Attr("members"), Attr("deputies"), No, No),
    ),
    (
        "report.finding",
        plan(Attr("score"), No, Attr("finding"), No),
    ),
    (
        "report.figures",
        plan(
            Attr("net_sales"),
            Attr("equity_ratio"),
            Attr("source"),
            Attr("auditor_remark"),
        ),
    ),
    ("name.changed", plan(No, No, Attr("naming_type"), No)),
    ("purpose.changed", plan(Attr("length"), No, No, No)),
    ("articles.changed", plan(No, No, No, No)),
    (
        "debt.registered",
        plan(
            Attr("amount"),
            Attr("cases"),
            Attr("debt_type"),
            Attr("case_type"),
        ),
    ),
    (
        "debt.present",
        plan(
            Attr("amount"),
            Attr("cases"),
            Attr("debt_type"),
            Attr("case_type"),
        ),
    ),
    (
        "debt.changed",
        plan(
            Attr("amount"),
            Diff("amount", "prev_amount"),
            Attr("debt_type"),
            Attr("case_type"),
        ),
    ),
    (
        "debt.cleared",
        plan(
            Attr("prev_amount"),
            Attr("prev_cases"),
            Attr("debt_type"),
            Attr("case_type"),
        ),
    ),
    (
        "enforcement.case_registered",
        plan(No, No, Attr("debt_type"), No),
    ),
    (
        "enforcement.case_under_enforcement",
        plan(Attr("original_debt"), No, Attr("case_type"), No),
    ),
    ("enforcement.case_closed", plan(No, No, Attr("cause"), No)),
    (
        "enforcement.case_partly_withdrawn",
        plan(Attr("recalled_amount"), No, No, No),
    ),
    ("enforcement.case_cancelled", plan(No, No, No, No)),
    (
        "enforcement.no_assets_report",
        plan(No, No, Attr("reason"), No),
    ),
    ("enforcement.decision_cancelled", plan(No, No, No, No)),
    (
        "payment_order.applied",
        plan(
            Attr("amount"),
            No,
            Attr("notification"),
            Attr("applicant_is_person"),
        ),
    ),
    (
        "payment_order.verdict",
        plan(Attr("amount"), No, Attr("notification"), No),
    ),
    (
        "payment_order.changed",
        plan(Attr("amount"), No, Attr("contested"), No),
    ),
    ("payment_order.removed", plan(Attr("amount"), No, Attr("removal_cause"), No)),
    ("payroll.declared", plan(Attr("amount"), No, No, No)),
    (
        "payroll.corrected",
        plan(Attr("amount"), Diff("amount", "prev_amount"), No, No),
    ),
    ("tax.late_fee_return", plan(Attr("amount"), No, No, No)),
    ("tax.late_fee_income", plan(Attr("amount"), No, No, No)),
    ("tax.surcharge", plan(Attr("amount"), No, Attr("kind"), No)),
    (
        "tax.deferral",
        plan(Attr("outstanding"), Attr("granted"), No, No),
    ),
    (
        "tax.registration",
        // num = the first recall reason code (514–528), the misconduct test of the question
        plan(
            FirstCode("reasons"),
            No,
            Attr("fskatt"),
            Pair("vat", "employer"),
        ),
    ),
    ("industry.code", plan(Attr("rank"), No, Attr("sni"), No)),
    ("permit.serving_granted", plan(No, No, Attr("category"), No)),
    ("permit.serving_ended", plan(LagDays("granted"), No, No, No)),
    ("role.started", plan(No, No, Attr("code"), No)),
    ("role.ended", plan(No, No, Attr("code"), No)),
    ("ownership.owner_added", plan(No, No, Attr("extent"), No)),
    ("ownership.owner_removed", plan(No, No, Attr("extent"), No)),
];

/// relation attributes by kind prefix: roles keep the role code, ownership its extent band,
/// addresses the address type, group edges the share
pub fn rel_plan(kind: &str) -> AttrPlan {
    if kind.starts_with("role.") {
        plan(No, No, Attr("code"), No)
    } else if kind.starts_with("ownership.") {
        plan(No, No, Attr("extent"), No)
    } else if kind == "group.parent" {
        plan(Attr("share"), No, No, No)
    } else if kind == "address.registered" {
        plan(No, No, Attr("type"), No)
    } else {
        plan(No, No, No, No)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Dict {
    pub event_kinds: Vec<String>,
    pub rel_kinds: Vec<String>,
    /// `key=value` strings; code 0 is "none"
    pub cats: Vec<String>,
    /// the columns of `figures.bin`
    #[serde(default)]
    pub figure_names: Vec<String>,
}

impl Dict {
    pub fn event_kind(&self, name: &str) -> Option<u16> {
        self.event_kinds
            .iter()
            .position(|k| k == name)
            .and_then(|i| u16::try_from(i).ok())
    }

    pub fn rel_kind(&self, name: &str) -> Option<u16> {
        self.rel_kinds
            .iter()
            .position(|k| k == name)
            .and_then(|i| u16::try_from(i).ok())
    }

    pub fn cat(&self, key_value: &str) -> Option<u16> {
        self.cats
            .iter()
            .position(|k| k == key_value)
            .and_then(|i| u16::try_from(i).ok())
    }

    /// the plan of a kind id (None past the closed table)
    pub fn event_plan(kind: u16) -> Option<&'static AttrPlan> {
        EVENT_KINDS.get(usize::from(kind)).map(|(_, p)| p)
    }
}

struct Interner {
    map: HashMap<String, u16>,
    list: Vec<String>,
}

impl Interner {
    fn new() -> Self {
        let mut i = Interner {
            map: HashMap::new(),
            list: Vec::new(),
        };
        i.code("");
        i
    }

    fn with(list: &[&str]) -> Self {
        let mut i = Interner::new();
        i.list.clear();
        i.map.clear();
        for s in list {
            i.code(s);
        }
        i
    }

    fn code(&mut self, s: &str) -> u16 {
        if let Some(&c) = self.map.get(s) {
            return c;
        }
        let c = u16::try_from(self.list.len()).unwrap_or(u16::MAX);
        self.map.insert(s.to_owned(), c);
        self.list.push(s.to_owned());
        c
    }
}

fn num_of(v: Option<&Value>) -> Option<f32> {
    match v? {
        Value::Number(n) => n.as_f64().map(|f| f as f32),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse::<f64>().ok().map(|f| f as f32),
        _ => None,
    }
}

fn cat_text(key: &str, v: Option<&Value>) -> Option<String> {
    match v? {
        Value::Null => None,
        Value::String(s) => Some(format!("{key}={s}")),
        Value::Number(n) => Some(format!("{key}={n}")),
        Value::Bool(b) => Some(format!("{key}={b}")),
        other => Some(format!("{key}={other}")),
    }
}

fn numeric(spec: Spec, attrs: &serde_json::Map<String, Value>, valid: Day) -> f32 {
    match spec {
        No | Pair(..) => 0.0,
        Attr(k) => num_of(attrs.get(k)).unwrap_or(0.0),
        LagDays(k) => attrs
            .get(k)
            .and_then(Value::as_str)
            .and_then(parse_ymd)
            .map_or(0.0, |d| (valid - d) as f32),
        Diff(a, b) => num_of(attrs.get(a)).unwrap_or(0.0) - num_of(attrs.get(b)).unwrap_or(0.0),
        FirstCode(k) => attrs
            .get(k)
            .and_then(|v| match v {
                Value::String(s) => s
                    .split(',')
                    .next()
                    .and_then(|c| c.trim().parse::<f32>().ok()),
                other => num_of(Some(other)),
            })
            .unwrap_or(0.0),
    }
}

fn categorical(spec: Spec, attrs: &serde_json::Map<String, Value>) -> Option<String> {
    match spec {
        No | LagDays(_) | Diff(..) | FirstCode(_) => None,
        Attr(k) => cat_text(k, attrs.get(k)),
        Pair(a, b) => match (cat_text(a, attrs.get(a)), cat_text(b, attrs.get(b))) {
            (None, None) => None,
            (x, y) => Some(format!(
                "{},{}",
                x.unwrap_or_default(),
                y.unwrap_or_default()
            )),
        },
    }
}

#[derive(Deserialize)]
struct EventLine<'a> {
    c: i32,
    #[serde(borrow)]
    k: &'a str,
    #[serde(borrow)]
    t: &'a str,
    #[serde(borrow)]
    ka: &'a str,
    est: bool,
    a: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct EndpointLine<'a> {
    #[serde(borrow)]
    t: &'a str,
    id: Value,
}

#[derive(Deserialize)]
struct RelLine<'a> {
    #[serde(borrow)]
    a: EndpointLine<'a>,
    #[serde(borrow)]
    k: &'a str,
    #[serde(borrow)]
    b: EndpointLine<'a>,
    #[serde(borrow)]
    from: Option<&'a str>,
    #[serde(borrow)]
    to: Option<&'a str>,
    #[serde(borrow)]
    ka: &'a str,
    est: bool,
    x: serde_json::Map<String, Value>,
}

/// a parsed event before its categories are interned
struct Parsed<R> {
    rec: R,
    kind: String,
    cat: Option<String>,
    cat2: Option<String>,
    /// a `report.figures` event's row, `figure_attrs()` order, NaN when absent
    figs: Option<Vec<f32>>,
}

fn bad(line: u64, msg: impl Into<String>) -> Error {
    Error::Record {
        line,
        msg: msg.into(),
    }
}

fn parse_event(
    line: &[u8],
    plans: &HashMap<&str, AttrPlan>,
    figure_attrs: &[&str],
) -> Result<Parsed<EventRec>, String> {
    let l: EventLine = serde_json::from_slice(line).map_err(|e| e.to_string())?;
    let valid = parse_ymd(l.t).ok_or("bad valid time")?;
    let known = parse_ymd(l.ka).ok_or("bad known_at")?;
    let p = plans.get(l.k).copied().unwrap_or(plan(No, No, No, No));
    let figs = (l.k == "report.figures").then(|| {
        figure_attrs
            .iter()
            .map(|a| num_of(l.a.get(*a)).unwrap_or(f32::NAN))
            .collect::<Vec<f32>>()
    });
    Ok(Parsed {
        rec: EventRec {
            company: l.c,
            kind: 0,
            est: l.est,
            valid,
            known,
            num: numeric(p.num, &l.a, valid),
            num2: numeric(p.num2, &l.a, valid),
            cat: 0,
            cat2: 0,
            extra: NO_EXTRA,
        },
        kind: l.k.to_owned(),
        cat: categorical(p.cat, &l.a),
        cat2: categorical(p.cat2, &l.a),
        figs,
    })
}

/// the address key is a 32-hex hash: its first 16 hex digits are the id. other strings: FNV-1a
pub fn address_id(key: &str) -> i64 {
    if key.len() >= 16
        && let Ok(v) = u64::from_str_radix(&key[..16], 16)
    {
        return v as i64;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h as i64
}

fn endpoint(e: &EndpointLine) -> Result<(EndpointType, i64), String> {
    match e.t {
        "company" => Ok((EndpointType::Company, e.id.as_i64().ok_or("company id")?)),
        "person" => Ok((EndpointType::Person, e.id.as_i64().ok_or("person id")?)),
        "address" => Ok((
            EndpointType::Address,
            address_id(e.id.as_str().ok_or("address key")?),
        )),
        other => Err(format!("endpoint type {other}")),
    }
}

fn parse_rel(line: &[u8]) -> Result<Parsed<RelRec>, String> {
    let l: RelLine = serde_json::from_slice(line).map_err(|e| e.to_string())?;
    let (at, aid) = endpoint(&l.a)?;
    if at != EndpointType::Company {
        return Err("subject is not a company".into());
    }
    let (bt, bid) = endpoint(&l.b)?;
    let known = parse_ymd(l.ka).ok_or("bad known_at")?;
    let from = match l.from {
        Some(s) => parse_ymd(s).ok_or("bad from")?,
        None => DAY_MIN,
    };
    let to = match l.to {
        Some(s) => parse_ymd(s).ok_or("bad to")?,
        None => DAY_MAX,
    };
    let p = rel_plan(l.k);
    Ok(Parsed {
        rec: RelRec {
            company: i32::try_from(aid).map_err(|_| "company id range")?,
            kind: 0,
            b_type: bt,
            est: l.est,
            b_id: bid,
            from,
            to,
            known,
            num: numeric(p.num, &l.x, from),
            cat: 0,
        },
        kind: l.k.to_owned(),
        cat: categorical(p.cat, &l.x),
        cat2: None,
        figs: None,
    })
}

/// reads a JSON-lines file in large chunks, parsing each chunk's lines in parallel; `sink` runs
/// sequentially per chunk (interning, pushing)
fn for_each_chunk<T: Send>(
    path: &Path,
    parse: impl Fn(&[u8]) -> Result<T, String> + Sync,
    mut sink: impl FnMut(Vec<T>) -> Result<(), Error>,
) -> Result<u64, Error> {
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut filled = 0usize;
    let mut line_no = 0u64;
    loop {
        let n = file.read(&mut buf[filled..])?;
        let eof = n == 0;
        filled += n;
        let end = if eof {
            filled
        } else {
            match buf[..filled].iter().rposition(|&b| b == b'\n') {
                Some(i) => i + 1,
                None if filled == buf.len() => {
                    buf.resize(buf.len() * 2, 0);
                    continue;
                }
                None => continue,
            }
        };
        if end > 0 {
            let lines: Vec<(u64, &[u8])> = buf[..end]
                .split(|&b| b == b'\n')
                .filter(|l| !l.is_empty() && *l != b"\r")
                .enumerate()
                .map(|(i, l)| (line_no + i as u64 + 1, l.strip_suffix(b"\r").unwrap_or(l)))
                .collect();
            let count = lines.len() as u64;
            let parsed: Result<Vec<T>, Error> = lines
                .par_iter()
                .map(|(no, l)| parse(l).map_err(|m| bad(*no, m)))
                .collect();
            sink(parsed?)?;
            line_no += count;
        }
        if eof {
            break;
        }
        buf.copy_within(end..filled, 0);
        filled -= end;
    }
    Ok(line_no)
}

#[derive(Debug, Default, Serialize)]
pub struct Stats {
    pub events: u64,
    pub relations: u64,
    pub event_kinds: usize,
    pub rel_kinds: usize,
    pub cats: usize,
    pub seconds: f64,
}

/// `events.jsonl` + `relations.jsonl` in `risk_dir` → `events.bin`, `relations.bin`, `dict.json`
pub fn build(risk_dir: &Path, cache_dir: &Path) -> Result<Stats, Error> {
    let started = Instant::now();
    std::fs::create_dir_all(cache_dir)?;
    let plans: HashMap<&str, AttrPlan> = EVENT_KINDS.iter().map(|(k, p)| (*k, *p)).collect();
    let mut kinds = Interner::with(&EVENT_KINDS.iter().map(|(k, _)| *k).collect::<Vec<_>>());
    let mut cats = Interner::new();
    let figure_names = figure_attrs();
    let width = figure_names.len();
    let mut figures: Vec<f32> = Vec::new();
    let mut events: Vec<EventRec> = Vec::with_capacity(52_000_000);
    let n_events = for_each_chunk(
        &risk_dir.join("events.jsonl"),
        |l| parse_event(l, &plans, &figure_names),
        |chunk| {
            for p in chunk {
                let mut rec = p.rec;
                rec.kind = kinds.code(&p.kind);
                rec.cat = p.cat.as_deref().map_or(0, |c| cats.code(c));
                rec.cat2 = p.cat2.as_deref().map_or(0, |c| cats.code(c));
                if let Some(f) = p.figs {
                    rec.extra = u32::try_from(figures.len() / width).unwrap_or(NO_EXTRA);
                    figures.extend(f);
                }
                events.push(rec);
            }
            Ok(())
        },
    )?;
    tracing::info!(
        n_events,
        kinds = kinds.list.len(),
        figure_rows = figures.len() / width,
        "events parsed; sorting"
    );
    events.par_sort_unstable_by_key(|e| (e.company, e.valid, e.known));
    write_events(&cache_dir.join("events.bin"), &events)?;
    drop(events);
    write_figures(&cache_dir.join("figures.bin"), width, &figures)?;
    drop(figures);

    let mut rel_kinds = Interner::new();
    rel_kinds.list.clear();
    rel_kinds.map.clear();
    let mut rels: Vec<RelRec> = Vec::with_capacity(22_000_000);
    let n_rels = for_each_chunk(&risk_dir.join("relations.jsonl"), parse_rel, |chunk| {
        for p in chunk {
            let mut rec = p.rec;
            rec.kind = rel_kinds.code(&p.kind);
            rec.cat = p.cat.as_deref().map_or(0, |c| cats.code(c));
            rels.push(rec);
        }
        Ok(())
    })?;
    tracing::info!(
        n_rels,
        kinds = rel_kinds.list.len(),
        "relations parsed; sorting"
    );
    rels.par_sort_unstable_by_key(|r| (r.company, r.from, r.known));
    write_rels(&cache_dir.join("relations.bin"), &rels)?;

    let dict = Dict {
        event_kinds: kinds.list,
        rel_kinds: rel_kinds.list,
        cats: cats.list,
        figure_names: figure_names.iter().map(|s| (*s).to_owned()).collect(),
    };
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(cache_dir.join("dict.json"))?),
        &dict,
    )?;
    Ok(Stats {
        events: n_events,
        relations: n_rels,
        event_kinds: dict.event_kinds.len(),
        rel_kinds: dict.rel_kinds.len(),
        cats: dict.cats.len(),
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn write_events(path: &Path, events: &[EventRec]) -> Result<(), Error> {
    let mut out = BufWriter::with_capacity(8 << 20, File::create(path)?);
    out.write_all(EVENT_MAGIC)?;
    out.write_all(&(events.len() as u64).to_le_bytes())?;
    let mut b = [0u8; EVENT_REC_SIZE];
    for e in events {
        b[0..4].copy_from_slice(&e.company.to_le_bytes());
        b[4..6].copy_from_slice(&e.kind.to_le_bytes());
        b[6] = u8::from(e.est);
        b[7] = 0;
        b[8..12].copy_from_slice(&e.valid.to_le_bytes());
        b[12..16].copy_from_slice(&e.known.to_le_bytes());
        b[16..20].copy_from_slice(&e.num.to_le_bytes());
        b[20..24].copy_from_slice(&e.num2.to_le_bytes());
        b[24..26].copy_from_slice(&e.cat.to_le_bytes());
        b[26..28].copy_from_slice(&e.cat2.to_le_bytes());
        b[28..32].copy_from_slice(&e.extra.to_le_bytes());
        out.write_all(&b)?;
    }
    out.flush()?;
    Ok(())
}

fn write_rels(path: &Path, rels: &[RelRec]) -> Result<(), Error> {
    let mut out = BufWriter::with_capacity(8 << 20, File::create(path)?);
    out.write_all(REL_MAGIC)?;
    out.write_all(&(rels.len() as u64).to_le_bytes())?;
    let mut b = [0u8; REL_REC_SIZE];
    for r in rels {
        b[0..4].copy_from_slice(&r.company.to_le_bytes());
        b[4..6].copy_from_slice(&r.kind.to_le_bytes());
        b[6] = r.b_type as u8;
        b[7] = u8::from(r.est);
        b[8..16].copy_from_slice(&r.b_id.to_le_bytes());
        b[16..20].copy_from_slice(&r.from.to_le_bytes());
        b[20..24].copy_from_slice(&r.to.to_le_bytes());
        b[24..28].copy_from_slice(&r.known.to_le_bytes());
        b[28..32].copy_from_slice(&r.num.to_le_bytes());
        b[32..34].copy_from_slice(&r.cat.to_le_bytes());
        b[34..40].fill(0);
        out.write_all(&b)?;
    }
    out.flush()?;
    Ok(())
}

fn le_i32(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn le_i64(b: &[u8]) -> i64 {
    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn read_body(path: &Path, magic: &[u8; 8], rec_size: usize) -> Result<(Vec<u8>, usize), Error> {
    let name = path.display().to_string();
    let data = std::fs::read(path)?;
    if data.len() < 16 || &data[..8] != magic {
        return Err(Error::Cache(name, "bad magic".into()));
    }
    let n = usize::try_from(u64::from_le_bytes(
        data[8..16]
            .try_into()
            .map_err(|_| Error::Cache(name.clone(), "header".into()))?,
    ))
    .map_err(|_| Error::Cache(name.clone(), "count".into()))?;
    if data.len() != 16 + n * rec_size {
        return Err(Error::Cache(
            name,
            format!("expected {n} records, size mismatch"),
        ));
    }
    Ok((data, n))
}

pub fn load_events(cache_dir: &Path) -> Result<Vec<EventRec>, Error> {
    let (data, n) = read_body(&cache_dir.join("events.bin"), EVENT_MAGIC, EVENT_REC_SIZE)?;
    let mut out = Vec::with_capacity(n);
    for b in data[16..].as_chunks::<EVENT_REC_SIZE>().0 {
        out.push(EventRec {
            company: le_i32(&b[0..4]),
            kind: le_u16(&b[4..6]),
            est: b[6] != 0,
            valid: le_i32(&b[8..12]),
            known: le_i32(&b[12..16]),
            num: le_f32(&b[16..20]),
            num2: le_f32(&b[20..24]),
            cat: le_u16(&b[24..26]),
            cat2: le_u16(&b[26..28]),
            extra: le_u32(&b[28..32]),
        });
    }
    Ok(out)
}

fn write_figures(path: &Path, width: usize, data: &[f32]) -> Result<(), Error> {
    let mut out = BufWriter::with_capacity(8 << 20, File::create(path)?);
    out.write_all(FIGURES_MAGIC)?;
    out.write_all(&((data.len() / width.max(1)) as u64).to_le_bytes())?;
    out.write_all(&(width as u64).to_le_bytes())?;
    let mut buf = Vec::with_capacity(1 << 20);
    for chunk in data.chunks(1 << 18) {
        buf.clear();
        for v in chunk {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        out.write_all(&buf)?;
    }
    out.flush()?;
    Ok(())
}

/// (width, rows × width) of `figures.bin`
pub fn load_figures(cache_dir: &Path) -> Result<(usize, Vec<f32>), Error> {
    let path = cache_dir.join("figures.bin");
    let name = path.display().to_string();
    let data = std::fs::read(&path)?;
    if data.len() < 24 || &data[..8] != FIGURES_MAGIC {
        return Err(Error::Cache(name, "bad magic".into()));
    }
    let rows = usize::try_from(u64::from_le_bytes(
        data[8..16]
            .try_into()
            .map_err(|_| Error::Cache(name.clone(), "header".into()))?,
    ))
    .map_err(|_| Error::Cache(name.clone(), "rows".into()))?;
    let width = usize::try_from(u64::from_le_bytes(
        data[16..24]
            .try_into()
            .map_err(|_| Error::Cache(name.clone(), "header".into()))?,
    ))
    .map_err(|_| Error::Cache(name.clone(), "width".into()))?;
    if data.len() != 24 + rows * width * 4 {
        return Err(Error::Cache(name, "size mismatch".into()));
    }
    let out = data[24..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((width, out))
}

pub fn load_relations(cache_dir: &Path) -> Result<Vec<RelRec>, Error> {
    let (data, n) = read_body(&cache_dir.join("relations.bin"), REL_MAGIC, REL_REC_SIZE)?;
    let mut out = Vec::with_capacity(n);
    for b in data[16..].as_chunks::<REL_REC_SIZE>().0 {
        out.push(RelRec {
            company: le_i32(&b[0..4]),
            kind: le_u16(&b[4..6]),
            b_type: match b[6] {
                1 => EndpointType::Person,
                2 => EndpointType::Address,
                _ => EndpointType::Company,
            },
            est: b[7] != 0,
            b_id: le_i64(&b[8..16]),
            from: le_i32(&b[16..20]),
            to: le_i32(&b[20..24]),
            known: le_i32(&b[24..28]),
            num: le_f32(&b[28..32]),
            cat: le_u16(&b[32..34]),
        });
    }
    Ok(out)
}

pub fn load_dict(cache_dir: &Path) -> Result<Dict, Error> {
    Ok(serde_json::from_reader(File::open(
        cache_dir.join("dict.json"),
    )?)?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn plans() -> HashMap<&'static str, AttrPlan> {
        EVENT_KINDS.iter().map(|(k, p)| (*k, *p)).collect()
    }

    #[test]
    fn event_line_maps_attributes_by_plan() {
        let line = br#"{"c":1001,"k":"debt.changed","t":"2025-01-10","ka":"2025-01-12","est":false,"s":"kronofogden","a":{"case_type":0,"debt_type":20,"amount":250000,"cases":13,"prev_amount":240000,"prev_cases":12}}"#;
        let p = parse_event(line, &plans(), &figure_attrs()).unwrap();
        assert_eq!(p.kind, "debt.changed");
        assert_eq!(p.rec.company, 1001);
        assert_eq!(p.rec.known - p.rec.valid, 2);
        assert_eq!(p.rec.num, 250_000.0);
        assert_eq!(p.rec.num2, 10_000.0);
        assert_eq!(p.cat.as_deref(), Some("debt_type=20"));
        assert_eq!(p.cat2.as_deref(), Some("case_type=0"));

        let line = br#"{"c":1,"k":"report.registered","t":"2025-03-31","ka":"2025-03-31","est":false,"s":"bolagsverket","a":{"period_start":"2023-09-01","period_end":"2024-08-31","interim":false,"ruleset":"K2"}}"#;
        let p = parse_event(line, &plans(), &figure_attrs()).unwrap();
        assert_eq!(p.rec.num, 212.0, "filing lag in days");
        assert_eq!(p.cat.as_deref(), Some("ruleset=K2"));
        assert_eq!(p.cat2.as_deref(), Some("interim=false"));

        let line = br#"{"c":1,"k":"tax.registration","t":"2025-03-25","ka":"2025-03-25","est":true,"s":"skatteverket","a":{"fskatt":"F","vat":true,"employer":true}}"#;
        let p = parse_event(line, &plans(), &figure_attrs()).unwrap();
        assert_eq!(p.cat2.as_deref(), Some("vat=true,employer=true"));

        let line = br#"{"c":1,"k":"never.seen","t":"2025-03-25","ka":"2025-03-25","est":true,"s":"x","a":{"q":1}}"#;
        let p = parse_event(line, &plans(), &figure_attrs()).unwrap();
        assert_eq!(p.rec.num, 0.0);
        assert!(p.cat.is_none());
    }

    #[test]
    fn figures_event_fills_a_row_in_attribute_order() {
        let line = br#"{"c":1,"k":"report.figures","t":"2024-12-31","ka":"2025-07-02","est":false,"s":"bolagsverket","a":{"source":"digital","precision":"exact","months":12,"consolidated":false,"interim":false,"auditor_remark":false,"net_sales":1500000,"equity_ratio":-0.2734,"net_sales_change":-0.2918}}"#;
        let attrs = figure_attrs();
        let p = parse_event(line, &plans(), &attrs).unwrap();
        let f = p.figs.unwrap();
        assert_eq!(f.len(), attrs.len());
        assert_eq!(
            f[attrs.iter().position(|a| *a == "net_sales").unwrap()],
            1_500_000.0
        );
        assert_eq!(
            f[attrs.iter().position(|a| *a == "equity_ratio").unwrap()],
            -0.2734
        );
        assert_eq!(f[attrs.iter().position(|a| *a == "months").unwrap()], 12.0);
        assert!(f[attrs.iter().position(|a| *a == "cash").unwrap()].is_nan());
        assert_eq!(p.rec.num, 1_500_000.0);
        assert_eq!(p.cat.as_deref(), Some("source=digital"));
        assert_eq!(p.cat2.as_deref(), Some("auditor_remark=false"));
    }

    #[test]
    fn relation_line_parses_endpoints_and_intervals() {
        let line = br#"{"a":{"t":"company","id":1002},"k":"address.registered","b":{"t":"address","id":"0123456789abcdef0123456789abcdef"},"from":"2025-06-23","to":"2025-06-30","ka":"2025-06-23","est":false,"x":{"type":0}}"#;
        let p = parse_rel(line).unwrap();
        assert_eq!(p.rec.b_type, EndpointType::Address);
        assert_eq!(p.rec.b_id, address_id("0123456789abcdef0123456789abcdef"));
        assert_eq!(p.rec.to - p.rec.from, 7);
        assert_eq!(p.cat.as_deref(), Some("type=0"));
        let line = br#"{"a":{"t":"company","id":1003},"k":"group.parent","b":{"t":"company","id":1004},"from":null,"to":null,"ka":"2025-06-23","est":false,"x":{"share":100}}"#;
        let p = parse_rel(line).unwrap();
        assert_eq!(p.rec.from, DAY_MIN);
        assert_eq!(p.rec.to, DAY_MAX);
        assert_eq!(p.rec.num, 100.0);
        assert!(p.rec.active_on(0));
    }

    #[test]
    fn records_round_trip_through_the_binary_files() {
        let dir = std::env::temp_dir().join(format!("risk-eval-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ev = vec![
            EventRec {
                company: 7,
                kind: 3,
                est: true,
                valid: -100,
                known: 5,
                num: 1.5,
                num2: -2.0,
                cat: 9,
                cat2: 0,
                extra: 3,
            },
            EventRec::default(),
        ];
        write_events(&dir.join("events.bin"), &ev).unwrap();
        assert_eq!(load_events(&dir).unwrap(), ev);
        write_figures(&dir.join("figures.bin"), 2, &[1.0, f32::NAN, 3.0, 4.0]).unwrap();
        let (w, f) = load_figures(&dir).unwrap();
        assert_eq!(w, 2);
        assert_eq!(f.len(), 4);
        assert!(f[1].is_nan());
        assert_eq!(f[3], 4.0);
        let rels = vec![RelRec {
            company: 1,
            kind: 2,
            b_type: EndpointType::Person,
            est: false,
            b_id: -99,
            from: DAY_MIN,
            to: 10,
            known: 3,
            num: 0.5,
            cat: 4,
        }];
        write_rels(&dir.join("relations.bin"), &rels).unwrap();
        assert_eq!(load_relations(&dir).unwrap(), rels);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn chunked_reader_keeps_lines_whole() {
        let dir = std::env::temp_dir().join(format!("risk-eval-chunk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        let text: String = (0..1000).map(|i| format!("{i}\n")).collect();
        std::fs::write(&path, text).unwrap();
        let mut seen = Vec::new();
        let n = for_each_chunk(
            &path,
            |l| {
                std::str::from_utf8(l)
                    .map_err(|e| e.to_string())?
                    .parse::<u32>()
                    .map_err(|e| e.to_string())
            },
            |chunk| {
                seen.extend(chunk);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(n, 1000);
        assert_eq!(seen, (0..1000).collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
