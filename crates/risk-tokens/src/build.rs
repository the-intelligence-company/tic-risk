//! `events.jsonl` + `relations.jsonl` + the context files → the token cache: `vocab.json`,
//! `records.bin`, `cells.bin`, `entities.bin`, `economy.bin`, `schema.json`, `build.json`. lines are
//! parsed in parallel per 64 MB chunk; codes are interned sequentially in the chunk's sink. kinds
//! and attributes outside the closed tables are counted, never silently dropped. records are
//! sorted by (entity, valid, known, kind); cells keep file order and are referenced by range.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use risk_eval::cache::address_id;
use risk_eval::store::{Economy, read_lives};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::dates::{DAY_MAX, DAY_MIN, Day, parse_ymd, to_naive};
use crate::fields::{self, DERIVED_LAG_DAYS, Treat, Unit};
use crate::record::{
    CELLS_MAGIC, Cell, CellType, ECONOMY_MAGIC, ENTITIES_MAGIC, EconomyRow, EntityRow,
    FLAG_DERIVED, FLAG_EST, FLAG_RELATION, PeerType, RECORDS_MAGIC, RecWriter, Record, schema_json,
    write_all,
};
use crate::texts;
use crate::vocab::{Ns, Vocab};

const CHUNK_BYTES: usize = 64 << 20;

#[derive(Debug, Default, Serialize)]
pub struct BuildStats {
    pub events: u64,
    pub relations: u64,
    pub derived: u64,
    pub records: u64,
    pub cells: u64,
    pub entities: u64,
    pub economy_rows: u64,
    pub vocab: usize,
    pub codes: usize,
    pub codes_without_text: usize,
    /// attributes met in the input that the closed table does not cover, by `kind.attr`
    pub skipped_attrs: BTreeMap<String, u64>,
    pub unknown_kinds: BTreeMap<String, u64>,
    /// date attributes marked `Future` that lay before the record's valid day
    pub past_future_dates: u64,
    /// numeric values dropped as implausible, by field (`plausible`)
    pub implausible_values: BTreeMap<String, u64>,
    pub seconds: f64,
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
    #[serde(borrow)]
    s: &'a str,
    a: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct EndpointLine<'a> {
    #[serde(borrow)]
    t: &'a str,
    id: Value,
}

#[derive(Deserialize)]
struct RelationLine<'a> {
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
    #[serde(default)]
    x: serde_json::Map<String, Value>,
}

/// a cell before its code is interned
struct CellDraft {
    field: u32,
    ctype: CellType,
    unit: Unit,
    value: f32,
    /// categorical: (field key, value)
    code: Option<(String, String)>,
}

/// (derived kind, its valid day, its cells)
type Derived = (u32, Day, Vec<CellDraft>);

struct RecordDraft {
    rec: Record,
    cells: Vec<CellDraft>,
}

#[derive(Default)]
struct LineOut {
    recs: Vec<RecordDraft>,
    skipped: Vec<String>,
    unknown_kind: Option<String>,
    past_future: u64,
    /// field keys of numeric values beyond any company (parsing errors, not facts)
    implausible: Vec<String>,
    derived: u64,
}

fn fnv(parts: &[&str]) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for b in p.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0x1f;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    let g = (h as u32) ^ ((h >> 32) as u32);
    if g == 0 { 1 } else { g }
}

/// a value beyond any company is a parsing error, not a fact: amounts above ten trillion,
/// negative or million-plus head counts, ratios beyond a thousandfold
pub fn plausible(unit: Unit, x: f32) -> bool {
    match unit {
        Unit::Sek => x.abs() <= 1e13,
        Unit::Count => (0.0..=1e6).contains(&x),
        Unit::Ratio => x.abs() <= 1e3,
        Unit::Percent => x.abs() <= 1e4,
        _ => true,
    }
}

/// an attribute value as the text a group key hashes (pseudonyms are keyed by it too)
pub fn value_text(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.trim().to_owned()).filter(|s| !s.is_empty()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        other => Some(other.to_string()),
    }
}

fn value_num(v: &Value) -> Option<f32> {
    match v {
        Value::Number(n) => n.as_f64().map(|f| f as f32),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok().map(|f| f as f32),
        _ => None,
    }
}

fn value_date(v: &Value) -> Option<Day> {
    v.as_str().and_then(parse_ymd)
}

/// `yyyymm` as a number or string → months before `valid`'s month
fn months_before(v: &Value, valid: Day) -> Option<f32> {
    let s = value_text(v)?;
    let digits: String = s.chars().filter(char::is_ascii_digit).collect();
    if digits.len() < 6 {
        return None;
    }
    let y: i32 = digits[..4].parse().ok()?;
    let m: u32 = digits[4..6].parse().ok()?;
    let period = chrono::NaiveDate::from_ymd_opt(y, m, 1)?;
    let d = to_naive(valid);
    let month_start = chrono::NaiveDate::from_ymd_opt(d.year(), d.month(), 1)?;
    Some(
        ((month_start.year() - period.year()) * 12
            + (i32::try_from(month_start.month()).ok()? - i32::try_from(period.month()).ok()?))
            as f32,
    )
}

use chrono::Datelike;

struct Ctx<'a> {
    vocab: &'a Vocab,
}

impl Ctx<'_> {
    fn cell(&self, field: u32, ctype: CellType, unit: Unit, value: f32) -> CellDraft {
        CellDraft {
            field,
            ctype,
            unit,
            value,
            code: None,
        }
    }

    fn cat(&self, field: u32, key: &str, value: String) -> CellDraft {
        CellDraft {
            field,
            ctype: CellType::Cat,
            unit: Unit::None,
            value: 0.0,
            code: Some((key.to_owned(), value)),
        }
    }

    /// the cells of one record's attributes; derived records and group parts come back too
    fn attrs(
        &self,
        kind: &str,
        attrs: &serde_json::Map<String, Value>,
        valid: Day,
        out: &mut LineOut,
    ) -> (Vec<CellDraft>, Vec<Derived>, u32) {
        let mut cells = Vec::with_capacity(attrs.len());
        let mut derived = Vec::new();
        let mut group_parts: Vec<(String, String)> = Vec::new();
        for (attr, v) in attrs {
            if v.is_null() {
                continue;
            }
            let Some((field_key, treat)) = fields::spec(kind, attr) else {
                out.skipped.push(format!("{kind}.{attr}"));
                continue;
            };
            let field = match treat {
                Treat::Group | Treat::Skip | Treat::Future(_) | Treat::Note | Treat::Period => 0,
                _ => match self.vocab.field(&field_key) {
                    Some(f) => f,
                    None => {
                        out.skipped.push(format!("{kind}.{attr}"));
                        continue;
                    }
                },
            };
            match treat {
                Treat::Num(unit) => {
                    if let Some(x) = value_num(v).filter(|x| x.is_finite()) {
                        if plausible(unit, x) {
                            cells.push(self.cell(field, CellType::Num, unit, x));
                        } else {
                            out.implausible.push(field_key.clone());
                        }
                    }
                }
                Treat::Cat => {
                    if let Some(s) = value_text(v) {
                        cells.push(self.cat(field, &field_key, s));
                    }
                }
                Treat::Bool => {
                    if let Some(x) = value_num(v) {
                        cells.push(self.cell(field, CellType::Bool, Unit::None, x.clamp(0.0, 1.0)));
                    }
                }
                Treat::Days | Treat::Deadline => {
                    if let Some(d) = value_date(v) {
                        cells.push(self.cell(
                            field,
                            CellType::Days,
                            Unit::Days,
                            (valid - d) as f32,
                        ));
                    }
                }
                Treat::Months => {
                    if let Some(m) = months_before(v, valid) {
                        cells.push(self.cell(field, CellType::Months, Unit::Months, m));
                    }
                }
                Treat::Period => {
                    if let Some(s) = value_text(v) {
                        if let Some(f) = self.vocab.field("tax.period") {
                            cells.push(self.cat(f, "tax.period", s.clone()));
                        }
                        if let (Some(f), Ok(y)) = (
                            self.vocab.field("tax.period_months"),
                            s.chars().take(4).collect::<String>().parse::<i32>(),
                        ) && let Some(year_end) = chrono::NaiveDate::from_ymd_opt(y, 12, 1)
                        {
                            let m = months_before(
                                &Value::String(year_end.format("%Y%m").to_string()),
                                valid,
                            );
                            if let Some(m) = m {
                                cells.push(self.cell(f, CellType::Months, Unit::Months, m));
                            }
                        }
                    }
                }
                Treat::Multi => {
                    // the registry's code column sometimes holds a date or a zero string
                    // instead of a code: not a code, no cell
                    if let Some(s) = value_text(v) {
                        for part in s
                            .split(',')
                            .map(str::trim)
                            .filter(|p| !p.is_empty())
                            .filter(|p| !(p.len() >= 8 && p.chars().all(|c| c.is_ascii_digit())))
                            .filter(|p| p.chars().any(|c| c != '0'))
                        {
                            cells.push(self.cat(field, &field_key, part.to_owned()));
                        }
                    }
                }
                Treat::Future(derived_kind) => {
                    if let Some(d) = value_date(v) {
                        if d >= valid {
                            if let Some(k) = self.vocab.kind(derived_kind) {
                                derived.push((k, d, Vec::new()));
                            }
                        } else {
                            out.past_future += 1;
                        }
                    }
                }
                Treat::Note => {
                    if let Some(d) = value_date(v) {
                        if let Some(f) = self.vocab.field("report.note_date") {
                            cells.push(self.cell(
                                f,
                                CellType::Days,
                                Unit::Days,
                                (valid - d) as f32,
                            ));
                        }
                    } else if let Some(x) = value_num(v).filter(|x| x.is_finite())
                        && let Some(f) = self.vocab.field("report.note_amount")
                    {
                        cells.push(self.cell(f, CellType::Num, Unit::Sek, x));
                    }
                }
                Treat::Group => {
                    if let Some(s) = value_text(v) {
                        group_parts.push((attr.clone(), s));
                    }
                }
                Treat::Skip => {}
            }
        }
        group_parts.sort();
        let group = if group_parts.is_empty() {
            0
        } else {
            let parts: Vec<&str> = std::iter::once(fields::family(kind))
                .chain(group_parts.iter().map(|(_, v)| v.as_str()))
                .collect();
            fnv(&parts)
        };
        // a derived record carries the parent's amount and cause cells so it reads on its own
        for (_, _, dcells) in &mut derived {
            for c in &cells {
                let key = self.vocab.key(c.field);
                if matches!(
                    key,
                    "payment_order.amount"
                        | "payment_order.removal_cause"
                        | "payment_order.contested"
                        | "payment_order.notification"
                        | "payment_order.applicant_is_person"
                ) {
                    dcells.push(CellDraft {
                        field: c.field,
                        ctype: c.ctype,
                        unit: c.unit,
                        value: c.value,
                        code: c.code.clone(),
                    });
                }
            }
        }
        (cells, derived, group)
    }

    fn event(&self, line: &[u8]) -> Result<LineOut, String> {
        let e: EventLine = serde_json::from_slice(line).map_err(|e| e.to_string())?;
        let mut out = LineOut::default();
        // a vocabulary without `debt.present` reads those debts as registered
        let k = match (e.k, self.vocab.kind("debt.present")) {
            ("debt.present", None) => "debt.registered",
            (k, _) => k,
        };
        let Some(kind) = self.vocab.kind(k) else {
            out.unknown_kind = Some(e.k.to_owned());
            return Ok(out);
        };
        let valid = parse_ymd(e.t).ok_or_else(|| format!("bad valid time {}", e.t))?;
        let known = parse_ymd(e.ka).ok_or_else(|| format!("bad known_at {}", e.ka))?;
        let source = if e.k.starts_with("permit.") {
            texts::source_index("folkhalsomyndigheten")
        } else {
            texts::source_index(e.s)
        };
        let (cells, derived, group) = self.attrs(e.k, &e.a, valid, &mut out);
        let rec = Record {
            entity: e.c,
            kind,
            valid,
            known,
            end: valid,
            cell_start: 0,
            cell_count: 0,
            source,
            flags: if e.est { FLAG_EST } else { 0 },
            peer_type: PeerType::None,
            peer_id: 0,
            group,
        };
        out.recs.push(RecordDraft { rec, cells });
        for (dkind, day, dcells) in derived {
            let late = day > known;
            out.derived += 1;
            out.recs.push(RecordDraft {
                rec: Record {
                    entity: e.c,
                    kind: dkind,
                    valid: day,
                    known: if day <= known {
                        known
                    } else {
                        day + DERIVED_LAG_DAYS
                    },
                    end: day,
                    cell_start: 0,
                    cell_count: 0,
                    source,
                    flags: FLAG_DERIVED | if late || e.est { FLAG_EST } else { 0 },
                    peer_type: PeerType::None,
                    peer_id: 0,
                    group,
                },
                cells: dcells,
            });
        }
        Ok(out)
    }

    fn relation(&self, line: &[u8]) -> Result<LineOut, String> {
        let r: RelationLine = serde_json::from_slice(line).map_err(|e| e.to_string())?;
        let mut out = LineOut::default();
        let Some(kind) = self.vocab.kind(r.k) else {
            out.unknown_kind = Some(r.k.to_owned());
            return Ok(out);
        };
        if r.a.t != "company" {
            return Ok(out);
        }
        let entity =
            r.a.id
                .as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .ok_or("subject id")?;
        let (peer_type, peer_id) = match r.b.t {
            "company" => (PeerType::Company, r.b.id.as_i64().ok_or("peer id")?),
            "person" => (PeerType::Person, r.b.id.as_i64().ok_or("peer id")?),
            "address" => (
                PeerType::Address,
                match &r.b.id {
                    Value::String(s) => address_id(s),
                    other => address_id(&other.to_string()),
                },
            ),
            _ => (PeerType::None, 0),
        };
        let known = parse_ymd(r.ka).ok_or_else(|| format!("bad known_at {}", r.ka))?;
        let from = r.from.and_then(parse_ymd);
        let to = r.to.and_then(parse_ymd).unwrap_or(DAY_MAX);
        // an interval without a start was first seen at its knowledge time: start there, estimated
        let (valid, est) = match from {
            Some(f) => (f, r.est),
            None => (known, true),
        };
        let (cells, _, _) = self.attrs(r.k, &r.x, valid, &mut out);
        out.recs.push(RecordDraft {
            rec: Record {
                entity,
                kind,
                valid,
                known,
                end: to.max(valid),
                cell_start: 0,
                cell_count: 0,
                source: texts::source_index("bolagsverket"),
                flags: FLAG_RELATION | if est { FLAG_EST } else { 0 },
                peer_type,
                peer_id,
                group: 0,
            },
            cells,
        });
        Ok(out)
    }
}

/// reads a JSON-lines file in large chunks, parsing each chunk's lines in parallel; `sink` runs
/// sequentially per chunk
pub fn for_each_chunk<T: Send>(
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
                .map(|(no, l)| parse(l).map_err(|msg| Error::Record { line: *no, msg }))
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

/// where the cells go: the cache file, or memory for one request
trait CellOut {
    fn count(&self) -> u64;
    fn put(&mut self, cells: &[Cell]) -> Result<(), Error>;
}

impl CellOut for RecWriter<Cell> {
    fn count(&self) -> u64 {
        self.count
    }

    fn put(&mut self, cells: &[Cell]) -> Result<(), Error> {
        self.extend(cells)
    }
}

impl CellOut for Vec<Cell> {
    fn count(&self) -> u64 {
        self.len() as u64
    }

    fn put(&mut self, cells: &[Cell]) -> Result<(), Error> {
        self.extend_from_slice(cells);
        Ok(())
    }
}

struct Sink<'a, C: CellOut> {
    vocab: &'a mut Vocab,
    cells: C,
    records: Vec<Record>,
    stats: BuildStats,
    cell_buf: Vec<Cell>,
}

impl<C: CellOut> Sink<'_, C> {
    fn take(&mut self, outs: Vec<LineOut>) -> Result<(), Error> {
        for o in outs {
            for s in o.skipped {
                *self.stats.skipped_attrs.entry(s).or_default() += 1;
            }
            if let Some(k) = o.unknown_kind {
                *self.stats.unknown_kinds.entry(k).or_default() += 1;
            }
            self.stats.past_future_dates += o.past_future;
            for f in o.implausible {
                *self.stats.implausible_values.entry(f).or_default() += 1;
            }
            self.stats.derived += o.derived;
            for d in o.recs {
                self.cell_buf.clear();
                for c in d.cells {
                    let code = match c.code {
                        Some((field, value)) => self.vocab.code(&field, &value),
                        None => 0,
                    };
                    self.cell_buf.push(Cell {
                        field: c.field,
                        code,
                        value: c.value,
                        ctype: c.ctype,
                        unit: c.unit as u8,
                    });
                }
                let mut rec = d.rec;
                rec.cell_start = u32::try_from(self.cells.count()).map_err(|_| {
                    Error::Other("more than 4 billion cells: widen cell_start".into())
                })?;
                rec.cell_count = u16::try_from(self.cell_buf.len()).unwrap_or(u16::MAX);
                self.cells
                    .put(&self.cell_buf[..usize::from(rec.cell_count)])?;
                self.records.push(rec);
            }
        }
        Ok(())
    }
}

fn economy(risk_dir: &Path, vocab: &mut Vocab, out: &Path) -> Result<u64, Error> {
    let eco = Economy::read(&risk_dir.join("economy.jsonl"))?;
    let mut rows = Vec::new();
    for s in eco
        .national
        .iter()
        .chain(eco.industry.iter().map(|(_, s)| s))
    {
        let field = vocab.series_field(&s.id, &s.name, Unit::Index);
        for &(period, published, value) in &s.rows {
            rows.push(EconomyRow {
                series: field,
                period,
                published,
                value,
            });
        }
    }
    rows.sort_by_key(|r| (r.series, r.period, r.published));
    write_all(&out.join("economy.bin"), ECONOMY_MAGIC, &rows)?;
    Ok(rows.len() as u64)
}

/// the training data's files in `risk_dir` → the token cache in `out`
pub fn build(risk_dir: &Path, form: &str, out: &Path) -> Result<BuildStats, Error> {
    let t0 = Instant::now();
    std::fs::create_dir_all(out)?;
    let mut vocab = Vocab::closed();
    let economy_rows = economy(risk_dir, &mut vocab, out)?;
    tracing::info!(
        series = vocab.count(Ns::Field),
        rows = economy_rows,
        "economy written"
    );

    let mut sink = Sink {
        vocab: &mut vocab,
        cells: RecWriter::create(&out.join("cells.bin"), CELLS_MAGIC)?,
        records: Vec::new(),
        stats: BuildStats::default(),
        cell_buf: Vec::new(),
    };
    let events = run_file(&risk_dir.join("events.jsonl"), &mut sink, true)?;
    tracing::info!(
        events,
        records = sink.records.len(),
        cells = sink.cells.count,
        secs = t0.elapsed().as_secs_f64(),
        "events parsed"
    );
    let relations = run_file(&risk_dir.join("relations.jsonl"), &mut sink, false)?;
    tracing::info!(
        relations,
        records = sink.records.len(),
        cells = sink.cells.count,
        secs = t0.elapsed().as_secs_f64(),
        "relations parsed"
    );
    let cells = sink.cells.finish()?;
    let mut records = std::mem::take(&mut sink.records);
    let mut stats = std::mem::take(&mut sink.stats);
    records.par_sort_unstable_by_key(|r| (r.entity, r.valid, r.known, r.kind));

    // entities: every life of the input, plus any record subject missing from the lives
    let lives = read_lives(&risk_dir.join(format!("lives_{form}.csv")))?;
    let entities = entity_rows(&lives, &records);

    write_all(&out.join("records.bin"), RECORDS_MAGIC, &records)?;
    write_all(&out.join("entities.bin"), ENTITIES_MAGIC, &entities)?;
    vocab.save(&out.join("vocab.json"))?;
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(out.join("schema.json"))?),
        &schema_json(),
    )?;

    stats.events = events;
    stats.relations = relations;
    stats.records = records.len() as u64;
    stats.cells = cells;
    stats.entities = entities.len() as u64;
    stats.economy_rows = economy_rows;
    stats.vocab = vocab.len();
    stats.codes = vocab.count(Ns::Code);
    stats.codes_without_text = vocab
        .entries
        .iter()
        .filter(|e| e.ns == Ns::Code)
        .filter(|e| {
            let (field, value) = e.key.split_once('=').unwrap_or((e.key.as_str(), ""));
            !texts::has_text(field, value)
        })
        .count();
    stats.seconds = t0.elapsed().as_secs_f64();
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(out.join("build.json"))?),
        &stats,
    )?;
    Ok(stats)
}

/// the entity index over sorted records: every life (its form, registration and end), plus any
/// record subject missing from the lives (form 255, open interval)
fn entity_rows(lives: &[risk_store::population::CompanyLife], records: &[Record]) -> Vec<EntityRow> {
    let mut entities: Vec<EntityRow> = lives
        .iter()
        .map(|l| EntityRow {
            entity: l.company_id,
            legal_form: u8::try_from(
                risk_store::population::LegalForm::ALL
                    .iter()
                    .position(|f| *f == l.legal_form)
                    .unwrap_or(255),
            )
            .unwrap_or(255),
            registered: l.registered.map_or(DAY_MIN, risk_eval::dates::from_naive),
            ceased: l.ceased.map_or(DAY_MAX, risk_eval::dates::from_naive),
            rec_start: 0,
            rec_count: 0,
        })
        .collect();
    entities.sort_by_key(|e| e.entity);
    entities.dedup_by_key(|e| e.entity);
    let mut extra: Vec<EntityRow> = Vec::new();
    let mut i = 0usize;
    let mut ei = 0usize;
    while i < records.len() {
        let e = records[i].entity;
        let mut j = i + 1;
        while j < records.len() && records[j].entity == e {
            j += 1;
        }
        while ei < entities.len() && entities[ei].entity < e {
            ei += 1;
        }
        if ei < entities.len() && entities[ei].entity == e {
            entities[ei].rec_start = i as u64;
            entities[ei].rec_count = (j - i) as u32;
        } else {
            extra.push(EntityRow {
                entity: e,
                legal_form: 255,
                registered: DAY_MIN,
                ceased: DAY_MAX,
                rec_start: i as u64,
                rec_count: (j - i) as u32,
            });
        }
        i = j;
    }
    entities.extend(extra);
    entities.sort_by_key(|e| e.entity);
    entities
}

/// one request's token streams, built in memory from the events and relations of a company and its
/// neighbours: the same line parsers, cell rules, derived records, sort order and entity index as
/// the cache build, over a vocabulary that starts as the training corpus's (so every code the model
/// knows keeps the id order it was trained with); a code never seen in training is added after it
/// and becomes `unknown` in the model's remap. nothing of it outlives the request
pub struct MemBuild {
    vocab: Vocab,
    cells: Vec<Cell>,
    records: Vec<Record>,
    stats: BuildStats,
}

impl MemBuild {
    pub fn new(corpus_vocab: Vocab) -> MemBuild {
        MemBuild { vocab: corpus_vocab, cells: Vec::new(), records: Vec::new(), stats: BuildStats::default() }
    }

    /// the vocabulary as it grows (the economy series fields are added before the context opens)
    pub fn vocab_mut(&mut self) -> &mut Vocab {
        &mut self.vocab
    }

    fn take(&mut self, outs: Vec<LineOut>) -> Result<(), Error> {
        let mut sink = Sink {
            vocab: &mut self.vocab,
            cells: std::mem::take(&mut self.cells),
            records: std::mem::take(&mut self.records),
            stats: std::mem::take(&mut self.stats),
            cell_buf: Vec::new(),
        };
        let r = sink.take(outs);
        self.cells = sink.cells;
        self.records = sink.records;
        self.stats = sink.stats;
        r
    }

    /// `events.jsonl` lines
    pub fn events<'a>(&mut self, lines: impl IntoIterator<Item = &'a str>) -> Result<u64, Error> {
        let mut outs = Vec::new();
        let mut n = 0u64;
        {
            let ctx = Ctx { vocab: &self.vocab };
            for l in lines {
                n += 1;
                outs.push(ctx.event(l.as_bytes()).map_err(|msg| Error::Record { line: n, msg })?);
            }
        }
        self.stats.events += n;
        self.take(outs)?;
        Ok(n)
    }

    /// `relations.jsonl` lines
    pub fn relations<'a>(&mut self, lines: impl IntoIterator<Item = &'a str>) -> Result<u64, Error> {
        let mut outs = Vec::new();
        let mut n = 0u64;
        {
            let ctx = Ctx { vocab: &self.vocab };
            for l in lines {
                n += 1;
                outs.push(ctx.relation(l.as_bytes()).map_err(|msg| Error::Record { line: n, msg })?);
            }
        }
        self.stats.relations += n;
        self.take(outs)?;
        Ok(n)
    }

    /// the streams: records sorted as the cache sorts them, the entity index over `lives`
    pub fn finish(self, lives: &[risk_store::population::CompanyLife]) -> Result<(crate::streams::Streams, BuildStats), Error> {
        let MemBuild { vocab, cells, mut records, mut stats } = self;
        records.sort_by_key(|r| (r.entity, r.valid, r.known, r.kind));
        let entities = entity_rows(lives, &records);
        stats.records = records.len() as u64;
        stats.cells = cells.len() as u64;
        stats.entities = entities.len() as u64;
        stats.vocab = vocab.len();
        let streams = crate::streams::Streams::from_parts(vocab, records, cells, entities)?;
        Ok((streams, stats))
    }
}

fn run_file<C: CellOut>(path: &Path, sink: &mut Sink<'_, C>, events: bool) -> Result<u64, Error> {
    // the chunk parser reads the vocabulary; the sink writes codes into it between chunks. the
    // borrow is split by taking a shared snapshot pointer for the parse of each chunk
    let mut lines = 0u64;
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut filled = 0usize;
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
            let parsed: Result<Vec<LineOut>, Error> = {
                let ctx = Ctx { vocab: sink.vocab };
                let chunk_lines: Vec<(u64, &[u8])> = buf[..end]
                    .split(|&b| b == b'\n')
                    .filter(|l| !l.is_empty() && *l != b"\r")
                    .enumerate()
                    .map(|(i, l)| (lines + i as u64 + 1, l.strip_suffix(b"\r").unwrap_or(l)))
                    .collect();
                lines += chunk_lines.len() as u64;
                chunk_lines
                    .par_iter()
                    .map(|(no, l)| {
                        let r = if events {
                            ctx.event(l)
                        } else {
                            ctx.relation(l)
                        };
                        r.map_err(|msg| Error::Record { line: *no, msg })
                    })
                    .collect()
            };
            sink.take(parsed?)?;
        }
        if eof {
            break;
        }
        buf.copy_within(end..filled, 0);
        filled -= end;
    }
    Ok(lines)
}

/// kept for tests and tools that read a JSON-lines file with the same chunking
pub fn count_lines(path: &Path) -> Result<u64, Error> {
    for_each_chunk(path, |_| Ok(()), |_| Ok(()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn ctx_vocab() -> Vocab {
        Vocab::closed()
    }

    #[test]
    fn event_lines_become_records_with_typed_cells() {
        let v = ctx_vocab();
        let ctx = Ctx { vocab: &v };
        let line = br#"{"c":42,"k":"debt.changed","t":"2025-01-10","ka":"2025-01-12","est":false,"s":"kronofogden","a":{"case_type":0,"debt_type":20,"amount":250000,"cases":13,"prev_amount":240000,"prev_cases":12}}"#;
        let out = ctx.event(line).unwrap();
        assert_eq!(out.recs.len(), 1);
        let d = &out.recs[0];
        assert_eq!(d.rec.entity, 42);
        assert_eq!(d.rec.kind, v.kind("debt.changed").unwrap());
        assert_eq!(d.rec.known - d.rec.valid, 2);
        assert_eq!(d.rec.source, texts::source_index("kronofogden"));
        assert_eq!(d.cells.len(), 6);
        let amount = d
            .cells
            .iter()
            .find(|c| c.field == v.field("debt.amount").unwrap())
            .unwrap();
        assert_eq!(amount.value, 250_000.0);
        assert_eq!(amount.unit, Unit::Sek);
        let ct = d
            .cells
            .iter()
            .find(|c| c.field == v.field("debt.case_type").unwrap())
            .unwrap();
        assert_eq!(
            ct.code.as_ref().unwrap(),
            &("debt.case_type".to_owned(), "0".to_owned())
        );
        assert!(out.skipped.is_empty());
    }

    #[test]
    fn future_dates_become_derived_records_and_codes_split() {
        let v = ctx_vocab();
        let ctx = Ctx { vocab: &v };
        let line = br#"{"c":7,"k":"payment_order.applied","t":"2024-12-18","ka":"2024-12-30","est":false,"s":"kronofogden","a":{"notification":"I","amount":339,"applicant_company":1005,"applicant_is_person":false,"withdrawn":"2025-01-20","dismissed":"2024-12-30"}}"#;
        let out = ctx.event(line).unwrap();
        assert_eq!(out.recs.len(), 3);
        assert_eq!(out.derived, 2);
        let withdrawn = out
            .recs
            .iter()
            .find(|d| d.rec.kind == v.kind("payment_order.withdrawn").unwrap())
            .unwrap();
        assert_eq!(withdrawn.rec.valid, parse_ymd("2025-01-20").unwrap());
        assert_eq!(
            withdrawn.rec.known,
            parse_ymd("2025-01-21").unwrap(),
            "after the file: day + lag"
        );
        assert!(withdrawn.rec.flags & FLAG_EST != 0);
        assert!(withdrawn.rec.flags & FLAG_DERIVED != 0);
        assert!(
            withdrawn
                .cells
                .iter()
                .any(|c| v.key(c.field) == "payment_order.amount")
        );
        let dismissed = out
            .recs
            .iter()
            .find(|d| d.rec.kind == v.kind("payment_order.dismissed").unwrap())
            .unwrap();
        assert_eq!(
            dismissed.rec.known,
            parse_ymd("2024-12-30").unwrap(),
            "in the file: known with it"
        );
        assert!(dismissed.rec.flags & FLAG_EST == 0);
        assert!(
            out.recs[0]
                .cells
                .iter()
                .all(|c| v.key(c.field) != "payment_order.applicant_company")
        );

        let line = br#"{"c":7,"k":"case.received","t":"2025-06-19","ka":"2025-06-19","est":false,"s":"bolagsverket","a":{"codes":"LI,FRI","case_year":2025,"case_number":361872}}"#;
        let out = ctx.event(line).unwrap();
        let d = &out.recs[0];
        let codes: Vec<&str> = d
            .cells
            .iter()
            .filter_map(|c| c.code.as_ref().map(|(_, v)| v.as_str()))
            .collect();
        assert_eq!(codes, ["LI", "FRI"]);
        assert_ne!(d.rec.group, 0, "case key → group");
        let line2 = br#"{"c":7,"k":"case.terminated","t":"2025-07-01","ka":"2025-07-01","est":false,"s":"bolagsverket","a":{"outcome":"ceased","codes":"LI,FRI","case_year":2025,"case_number":361872}}"#;
        let out2 = ctx.event(line2).unwrap();
        assert_eq!(out2.recs[0].rec.group, d.rec.group, "same case, same group");
    }

    #[test]
    fn figures_periods_notes_and_unknowns() {
        let v = ctx_vocab();
        let ctx = Ctx { vocab: &v };
        let line = br#"{"c":1,"k":"report.figures","t":"2014-12-31","ka":"2015-07-31","est":true,"s":"bolagsverket","a":{"source":"vendor","precision":"thousand","months":12,"consolidated":false,"interim":false,"net_sales":34588000,"equity_ratio":0.42,"debt_equity":0.6,"employees":12,"mystery":5}}"#;
        let out = ctx.event(line).unwrap();
        let d = &out.recs[0];
        assert_eq!(out.skipped, vec!["report.figures.mystery".to_owned()]);
        let f = |k: &str| d.cells.iter().find(|c| v.key(c.field) == k).unwrap();
        assert_eq!(f("figure.net_sales").unit, Unit::Sek);
        assert_eq!(f("figure.equity_ratio").unit, Unit::Ratio);
        assert_eq!(f("figure.employees").unit, Unit::Count);
        assert_eq!(f("report.months").value, 12.0);
        assert_eq!(f("report.interim").ctype, CellType::Bool);
        assert!(d.rec.flags & FLAG_EST != 0);

        let line = br#"{"c":1,"k":"payroll.declared","t":"2022-03-31","ka":"2022-04-12","est":true,"s":"skatteverket","a":{"period":202202,"amount":225160,"version":405031580}}"#;
        let out = ctx.event(line).unwrap();
        let d = &out.recs[0];
        assert_eq!(d.cells.len(), 2, "version is dropped");
        let p = d
            .cells
            .iter()
            .find(|c| v.key(c.field) == "payroll.period")
            .unwrap();
        assert_eq!((p.ctype, p.value), (CellType::Months, 1.0));

        let line = br#"{"c":1,"k":"tax.late_fee_income","t":"2024-01-02","ka":"2024-02-01","est":true,"s":"skatteverket","a":{"period":"2021P4","amount":2500}}"#;
        let out = ctx.event(line).unwrap();
        let d = &out.recs[0];
        assert!(d.cells.iter().any(|c| {
            c.code
                .as_ref()
                .is_some_and(|(f, x)| f == "tax.period" && x == "2021P4")
        }));
        let m = d
            .cells
            .iter()
            .find(|c| v.key(c.field) == "tax.period_months")
            .unwrap();
        assert_eq!(m.value, 25.0, "2021-12 → 2024-01");

        let line = br#"{"c":1,"k":"report.finding","t":"2023-05-24","ka":"2023-05-31","est":true,"s":"bolagsverket","a":{"finding":"AGM_TOO_LATE","score":255,"note":"2022-06-30","period_end":"2021-12-31","report":1006}}"#;
        let out = ctx.event(line).unwrap();
        let d = &out.recs[0];
        assert!(d.cells.iter().any(|c| v.key(c.field) == "report.note_date"));
        assert_ne!(d.rec.group, 0);
        let line = br#"{"c":1,"k":"report.finding","t":"2023-05-24","ka":"2023-05-31","est":true,"s":"bolagsverket","a":{"finding":"FORBIDDEN_LOAN","score":200,"note":"125 000"}}"#;
        let out = ctx.event(line).unwrap();
        assert!(
            out.recs[0]
                .cells
                .iter()
                .all(|c| v.key(c.field) != "report.note_amount"),
            "a spaced number is not parsed as an amount"
        );

        let line = br#"{"c":1,"k":"no.such_kind","t":"2023-05-24","ka":"2023-05-31","est":true,"s":"bolagsverket","a":{}}"#;
        let out = ctx.event(line).unwrap();
        assert_eq!(out.unknown_kind.as_deref(), Some("no.such_kind"));
        assert!(out.recs.is_empty());
    }

    #[test]
    fn relation_lines_keep_peers_and_intervals() {
        let v = ctx_vocab();
        let ctx = Ctx { vocab: &v };
        let line = br#"{"a":{"t":"company","id":1007},"k":"role.board_member","b":{"t":"person","id":1008},"from":"2015-05-25","to":null,"ka":"2015-05-25","est":true,"x":{"code":"LE"}}"#;
        let out = ctx.relation(line).unwrap();
        let d = &out.recs[0];
        assert!(d.rec.is_relation());
        assert_eq!(d.rec.peer_type, PeerType::Person);
        assert_eq!(d.rec.peer_id, 1008);
        assert_eq!(d.rec.end, DAY_MAX);
        assert_eq!(d.rec.end_known(), DAY_MAX);
        assert_eq!(d.cells.len(), 1);
        let line = br#"{"a":{"t":"company","id":1002},"k":"address.registered","b":{"t":"address","id":"0123456789abcdef0123456789abcdef"},"from":"2025-06-23","to":"2025-06-30","ka":"2025-06-23","est":false,"x":{"type":0}}"#;
        let out = ctx.relation(line).unwrap();
        let d = &out.recs[0];
        assert_eq!(d.rec.peer_type, PeerType::Address);
        assert_eq!(
            d.rec.peer_id,
            address_id("0123456789abcdef0123456789abcdef")
        );
        assert_eq!(d.rec.end - d.rec.valid, 7);
        assert_eq!(d.rec.end_known(), d.rec.end);
        let line = br#"{"a":{"t":"company","id":1},"k":"group.parent","b":{"t":"company","id":2},"from":null,"to":null,"ka":"2025-06-23","est":false,"x":{"share":100}}"#;
        let out = ctx.relation(line).unwrap();
        let d = &out.recs[0];
        assert_eq!(
            d.rec.valid, d.rec.known,
            "no start: starts at its knowledge time"
        );
        assert!(d.rec.is_estimated());
    }

    #[test]
    fn implausible_values_are_dropped_and_counted() {
        let v = ctx_vocab();
        let ctx = Ctx { vocab: &v };
        let line = br#"{"c":1,"k":"report.figures","t":"2014-12-31","ka":"2015-07-31","est":true,"s":"bolagsverket","a":{"source":"scanned","precision":"exact","months":12,"consolidated":false,"interim":false,"net_sales":225169999864528896,"employees":-5,"equity_ratio":0.3}}"#;
        let out = ctx.event(line).unwrap();
        assert_eq!(
            out.implausible,
            vec!["figure.employees".to_owned(), "figure.net_sales".to_owned()],
            "attributes come in key order"
        );
        assert!(
            out.recs[0]
                .cells
                .iter()
                .any(|c| v.key(c.field) == "figure.equity_ratio")
        );
        assert!(
            out.recs[0]
                .cells
                .iter()
                .all(|c| v.key(c.field) != "figure.net_sales")
        );
        assert!(plausible(Unit::Sek, 5.0e12));
        assert!(!plausible(Unit::Ratio, 2000.0));
    }

    #[test]
    fn helpers() {
        assert_eq!(
            months_before(&Value::from(202212), parse_ymd("2023-03-15").unwrap()),
            Some(3.0)
        );
        assert_eq!(months_before(&Value::from("2022"), 0), None);
        assert_ne!(fnv(&["a", "b"]), fnv(&["ab"]));
        assert_ne!(fnv(&["x"]), 0);
    }
}
