//! the fixed-size little-endian layouts of the token cache: records with typed cells per entity,
//! the entity index, economy rows, and the materialised sample files (sample rows, index rows,
//! sequence tokens, slots). every file starts with an 8-byte magic and a 4-byte version and
//! record size; `schema_json` spells the layouts out as numpy dtypes.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::Error;
use crate::dates::{DAY_MAX, Day};

pub const RECORDS_MAGIC: &[u8; 8] = b"TICREC01";
pub const CELLS_MAGIC: &[u8; 8] = b"TICCEL01";
pub const ENTITIES_MAGIC: &[u8; 8] = b"TICENT01";
pub const ECONOMY_MAGIC: &[u8; 8] = b"TICECO01";
pub const SAMPLES_MAGIC: &[u8; 8] = b"TICSMP01";
pub const INDEX_MAGIC: &[u8; 8] = b"TICIDX01";
pub const TOKENS_MAGIC: &[u8; 8] = b"TICTOK01";
pub const SLOTS_MAGIC: &[u8; 8] = b"TICSLT01";
pub const HEADER_SIZE: usize = 16;
const VERSION: u32 = 1;

pub const FLAG_EST: u8 = 1;
pub const FLAG_RELATION: u8 = 2;
pub const FLAG_DERIVED: u8 = 4;
pub const FLAG_EXTRA: u8 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PeerType {
    #[default]
    None = 0,
    Company = 1,
    Person = 2,
    Address = 3,
}

impl PeerType {
    pub fn from_u8(v: u8) -> PeerType {
        match v {
            1 => PeerType::Company,
            2 => PeerType::Person,
            3 => PeerType::Address,
            _ => PeerType::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CellType {
    #[default]
    Num = 0,
    Cat = 1,
    Bool = 2,
    Days = 3,
    Months = 4,
}

impl CellType {
    pub fn from_u8(v: u8) -> CellType {
        match v {
            1 => CellType::Cat,
            2 => CellType::Bool,
            3 => CellType::Days,
            4 => CellType::Months,
            _ => CellType::Num,
        }
    }
}

pub trait Rec: Copy + Default {
    const SIZE: usize;
    fn write(&self, out: &mut [u8]);
    fn read(b: &[u8]) -> Self;
}

fn put_i32(b: &mut [u8], at: usize, v: i32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}
fn put_i64(b: &mut [u8], at: usize, v: i64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}
fn put_f32(b: &mut [u8], at: usize, v: f32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn get_i32(b: &[u8], at: usize) -> i32 {
    i32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn get_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn get_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}
fn get_i64(b: &[u8], at: usize) -> i64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    i64::from_le_bytes(a)
}
fn get_f32(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn get_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// one record: an event, a filed statement, a relation, or (in the extra space of a sample) a
/// derived state, context or query record. 48 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Record {
    pub entity: i32,
    pub kind: u32,
    pub valid: Day,
    pub known: Day,
    /// relations: the end day (`DAY_MAX` while open); events: the valid day
    pub end: Day,
    pub cell_start: u32,
    pub cell_count: u16,
    pub source: u8,
    pub flags: u8,
    pub peer_type: PeerType,
    pub peer_id: i64,
    /// same-case key (hash of the case identifier), 0 = none
    pub group: u32,
}

impl Record {
    pub fn is_relation(&self) -> bool {
        self.flags & FLAG_RELATION != 0
    }

    pub fn is_estimated(&self) -> bool {
        self.flags & FLAG_EST != 0
    }

    /// relations: the end's knowledge time, taken as the later of the end and the start's
    /// knowledge time
    pub fn end_known(&self) -> Day {
        if self.end == DAY_MAX {
            DAY_MAX
        } else {
            self.end.max(self.known)
        }
    }

    pub fn cells(&self) -> std::ops::Range<usize> {
        let s = self.cell_start as usize;
        s..s + usize::from(self.cell_count)
    }
}

impl Rec for Record {
    const SIZE: usize = 48;
    fn write(&self, b: &mut [u8]) {
        put_i32(b, 0, self.entity);
        put_u32(b, 4, self.kind);
        put_i32(b, 8, self.valid);
        put_i32(b, 12, self.known);
        put_i32(b, 16, self.end);
        put_u32(b, 20, self.cell_start);
        put_u16(b, 24, self.cell_count);
        b[26] = self.source;
        b[27] = self.flags;
        b[28] = self.peer_type as u8;
        b[29] = 0;
        b[30] = 0;
        b[31] = 0;
        put_i64(b, 32, self.peer_id);
        put_u32(b, 40, self.group);
        put_u32(b, 44, 0);
    }
    fn read(b: &[u8]) -> Self {
        Record {
            entity: get_i32(b, 0),
            kind: get_u32(b, 4),
            valid: get_i32(b, 8),
            known: get_i32(b, 12),
            end: get_i32(b, 16),
            cell_start: get_u32(b, 20),
            cell_count: get_u16(b, 24),
            source: b[26],
            flags: b[27],
            peer_type: PeerType::from_u8(b[28]),
            peer_id: get_i64(b, 32),
            group: get_u32(b, 40),
        }
    }
}

/// one typed attribute of a record. 16 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cell {
    pub field: u32,
    /// categorical cells: the code id; else 0
    pub code: u32,
    /// numeric, flag (1/0), days or months value
    pub value: f32,
    pub ctype: CellType,
    pub unit: u8,
}

impl Rec for Cell {
    const SIZE: usize = 16;
    fn write(&self, b: &mut [u8]) {
        put_u32(b, 0, self.field);
        put_u32(b, 4, self.code);
        put_f32(b, 8, self.value);
        b[12] = self.ctype as u8;
        b[13] = self.unit;
        b[14] = 0;
        b[15] = 0;
    }
    fn read(b: &[u8]) -> Self {
        Cell {
            field: get_u32(b, 0),
            code: get_u32(b, 4),
            value: get_f32(b, 8),
            ctype: CellType::from_u8(b[12]),
            unit: b[13],
        }
    }
}

/// one company with its record range. 32 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EntityRow {
    pub entity: i32,
    /// `LegalForm` code index (AB 0, HB 1, KB 2, EF 3); 255 unknown
    pub legal_form: u8,
    /// `DAY_MIN` when unknown
    pub registered: Day,
    /// `DAY_MAX` while alive
    pub ceased: Day,
    pub rec_start: u64,
    pub rec_count: u32,
}

impl Rec for EntityRow {
    const SIZE: usize = 32;
    fn write(&self, b: &mut [u8]) {
        put_i32(b, 0, self.entity);
        b[4] = self.legal_form;
        b[5] = 0;
        b[6] = 0;
        b[7] = 0;
        put_i32(b, 8, self.registered);
        put_i32(b, 12, self.ceased);
        put_u64(b, 16, self.rec_start);
        put_u32(b, 24, self.rec_count);
        put_u32(b, 28, 0);
    }
    fn read(b: &[u8]) -> Self {
        EntityRow {
            entity: get_i32(b, 0),
            legal_form: b[4],
            registered: get_i32(b, 8),
            ceased: get_i32(b, 12),
            rec_start: get_u64(b, 16),
            rec_count: get_u32(b, 24),
        }
    }
}

/// one economy value with its vintage. 16 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EconomyRow {
    /// the series' field id
    pub series: u32,
    pub period: Day,
    pub published: Day,
    pub value: f32,
}

impl Rec for EconomyRow {
    const SIZE: usize = 16;
    fn write(&self, b: &mut [u8]) {
        put_u32(b, 0, self.series);
        put_i32(b, 4, self.period);
        put_i32(b, 8, self.published);
        put_f32(b, 12, self.value);
    }
    fn read(b: &[u8]) -> Self {
        EconomyRow {
            series: get_u32(b, 0),
            period: get_i32(b, 4),
            published: get_i32(b, 8),
            value: get_f32(b, 12),
        }
    }
}

/// the horizons of the label block, in months
pub const HORIZONS: [u32; 4] = [3, 6, 12, 24];
/// `HorizonLabel::event` when the horizon is not complete at the data cut
pub const EVENT_INCOMPLETE: u8 = 255;
pub const SAMPLE_POSITIVE: u8 = 1;
pub const SAMPLE_COMPETING: u8 = 2;
pub const SAMPLE_CENSORED: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HorizonLabel {
    /// 0 none, 1 bankruptcy, 2 compulsory liquidation, 3 reconstruction, 4 deregistration,
    /// 255 incomplete
    pub event: u8,
    pub censored: u8,
    /// days from the as-of day to the event, or to the censoring, else -1
    pub days: i32,
}

/// one anchor: entity, as-of, split, the question's outcome, the label block. 64 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SampleRow {
    pub entity: i32,
    pub as_of: Day,
    /// 0 train, 1 test (held out)
    pub split: u8,
    /// SAMPLE_* bits for the plan's question
    pub flags: u8,
    /// the query target (position in `texts::QUERIES`)
    pub query: u8,
    pub horizon: u8,
    pub weight: f32,
    pub labels: [HorizonLabel; 4],
    /// the table's row number when the anchors come from a table, else the position
    pub row: u32,
}

impl Rec for SampleRow {
    const SIZE: usize = 64;
    fn write(&self, b: &mut [u8]) {
        put_i32(b, 0, self.entity);
        put_i32(b, 4, self.as_of);
        b[8] = self.split;
        b[9] = self.flags;
        b[10] = self.query;
        b[11] = self.horizon;
        put_f32(b, 12, self.weight);
        for (i, l) in self.labels.iter().enumerate() {
            let at = 16 + i * 8;
            b[at] = l.event;
            b[at + 1] = l.censored;
            b[at + 2] = 0;
            b[at + 3] = 0;
            put_i32(b, at + 4, l.days);
        }
        put_u32(b, 48, self.row);
        for x in &mut b[52..64] {
            *x = 0;
        }
    }
    fn read(b: &[u8]) -> Self {
        let mut labels = [HorizonLabel::default(); 4];
        for (i, l) in labels.iter_mut().enumerate() {
            let at = 16 + i * 8;
            *l = HorizonLabel {
                event: b[at],
                censored: b[at + 1],
                days: get_i32(b, at + 4),
            };
        }
        SampleRow {
            entity: get_i32(b, 0),
            as_of: get_i32(b, 4),
            split: b[8],
            flags: b[9],
            query: b[10],
            horizon: b[11],
            weight: get_f32(b, 12),
            labels,
            row: get_u32(b, 48),
        }
    }
}

pub const INDEX_FOCAL_TRUNCATED: u32 = 1;
pub const INDEX_NEIGHBOURS_TRUNCATED: u32 = 2;
pub const INDEX_BULK_ADDRESS: u32 = 4;

/// where one sample's tokens, targets, slots and extra records are. 48 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IndexRow {
    pub tok_start: u64,
    /// input tokens; the target tokens follow them
    pub tok_count: u32,
    pub target_count: u32,
    pub slot_start: u64,
    pub slot_count: u32,
    pub extra_start: u32,
    pub extra_count: u32,
    pub flags: u32,
}

impl Rec for IndexRow {
    const SIZE: usize = 48;
    fn write(&self, b: &mut [u8]) {
        put_u64(b, 0, self.tok_start);
        put_u32(b, 8, self.tok_count);
        put_u32(b, 12, self.target_count);
        put_u64(b, 16, self.slot_start);
        put_u32(b, 24, self.slot_count);
        put_u32(b, 28, self.extra_start);
        put_u32(b, 32, self.extra_count);
        put_u32(b, 36, self.flags);
        put_u64(b, 40, 0);
    }
    fn read(b: &[u8]) -> Self {
        IndexRow {
            tok_start: get_u64(b, 0),
            tok_count: get_u32(b, 8),
            target_count: get_u32(b, 12),
            slot_start: get_u64(b, 16),
            slot_count: get_u32(b, 24),
            extra_start: get_u32(b, 28),
            extra_count: get_u32(b, 32),
            flags: get_u32(b, 36),
        }
    }
}

/// `SeqTok::record` with this bit set indexes the sample's extra records, not `records.bin`
pub const EXTRA_BIT: u32 = 1 << 31;

/// one token of a materialised sample: a record reference with its slot, hop and path. 8 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SeqTok {
    pub record: u32,
    pub slot: u16,
    pub hop: u8,
    pub path: u8,
}

impl Rec for SeqTok {
    const SIZE: usize = 8;
    fn write(&self, b: &mut [u8]) {
        put_u32(b, 0, self.record);
        put_u16(b, 4, self.slot);
        b[6] = self.hop;
        b[7] = self.path;
    }
    fn read(b: &[u8]) -> Self {
        SeqTok {
            record: get_u32(b, 0),
            slot: get_u16(b, 4),
            hop: b[6],
            path: b[7],
        }
    }
}

pub const NO_VIA: u32 = u32::MAX;

/// one neighbour slot of a sample: the entity, how it was reached. 16 bytes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SlotRow {
    pub slot: u16,
    pub hop: u8,
    pub path: u8,
    pub entity: i32,
    /// the relation record (in `records.bin`) the neighbour came through; `NO_VIA` for the focal
    pub via: u32,
}

impl Rec for SlotRow {
    const SIZE: usize = 16;
    fn write(&self, b: &mut [u8]) {
        put_u16(b, 0, self.slot);
        b[2] = self.hop;
        b[3] = self.path;
        put_i32(b, 4, self.entity);
        put_u32(b, 8, self.via);
        put_u32(b, 12, 0);
    }
    fn read(b: &[u8]) -> Self {
        SlotRow {
            slot: get_u16(b, 0),
            hop: b[2],
            path: b[3],
            entity: get_i32(b, 4),
            via: get_u32(b, 8),
        }
    }
}

fn header(magic: &[u8; 8], size: usize) -> [u8; HEADER_SIZE] {
    let mut h = [0u8; HEADER_SIZE];
    h[..8].copy_from_slice(magic);
    h[8..12].copy_from_slice(&VERSION.to_le_bytes());
    h[12..16].copy_from_slice(&(size as u32).to_le_bytes());
    h
}

fn check_header(path: &Path, magic: &[u8; 8], size: usize, h: &[u8]) -> Result<(), Error> {
    let bad = |msg: &str| Error::Cache(path.display().to_string(), msg.to_owned());
    if h.len() < HEADER_SIZE || &h[..8] != magic {
        return Err(bad("wrong magic"));
    }
    if get_u32(h, 8) != VERSION {
        return Err(bad("wrong version"));
    }
    if get_u32(h, 12) as usize != size {
        return Err(bad("wrong record size"));
    }
    Ok(())
}

/// a streaming writer of one record type
pub struct RecWriter<T: Rec> {
    out: BufWriter<File>,
    buf: Vec<u8>,
    pub count: u64,
    _t: std::marker::PhantomData<T>,
}

impl<T: Rec> RecWriter<T> {
    pub fn create(path: &Path, magic: &[u8; 8]) -> Result<Self, Error> {
        let mut out = BufWriter::with_capacity(8 << 20, File::create(path)?);
        out.write_all(&header(magic, T::SIZE))?;
        Ok(RecWriter {
            out,
            buf: vec![0u8; T::SIZE],
            count: 0,
            _t: std::marker::PhantomData,
        })
    }

    pub fn push(&mut self, r: &T) -> Result<(), Error> {
        r.write(&mut self.buf);
        self.out.write_all(&self.buf)?;
        self.count += 1;
        Ok(())
    }

    pub fn extend(&mut self, rs: &[T]) -> Result<(), Error> {
        for r in rs {
            self.push(r)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64, Error> {
        self.out.flush()?;
        Ok(self.count)
    }
}

pub fn write_all<T: Rec>(path: &Path, magic: &[u8; 8], rows: &[T]) -> Result<(), Error> {
    let mut w = RecWriter::<T>::create(path, magic)?;
    w.extend(rows)?;
    w.finish()?;
    Ok(())
}

// `as_chunks` needs a const generic; `T::SIZE` is an associated const of a type parameter
#[allow(clippy::chunks_exact_to_as_chunks)]
pub fn read_all<T: Rec>(path: &Path, magic: &[u8; 8]) -> Result<Vec<T>, Error> {
    let f = File::open(path)?;
    let len = f.metadata()?.len() as usize;
    let mut r = BufReader::with_capacity(16 << 20, f);
    let mut h = [0u8; HEADER_SIZE];
    r.read_exact(&mut h)?;
    check_header(path, magic, T::SIZE, &h)?;
    let body = len.saturating_sub(HEADER_SIZE);
    if !body.is_multiple_of(T::SIZE) {
        return Err(Error::Cache(
            path.display().to_string(),
            "truncated".to_owned(),
        ));
    }
    let n = body / T::SIZE;
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; T::SIZE * 65536];
    let mut left = body;
    while left > 0 {
        let take = left.min(buf.len());
        r.read_exact(&mut buf[..take])?;
        for chunk in buf[..take].chunks_exact(T::SIZE) {
            out.push(T::read(chunk));
        }

        left -= take;
    }
    Ok(out)
}

/// the layouts as numpy structured dtypes
pub fn schema_json() -> serde_json::Value {
    serde_json::json!({
        "version": VERSION,
        "header": {"bytes": HEADER_SIZE, "fields": [["magic", "S8"], ["version", "<u4"], ["record_size", "<u4"]]},
        "records.bin": {"magic": "TICREC01", "size": Record::SIZE, "dtype": [
            ["entity", "<i4"], ["kind", "<u4"], ["valid", "<i4"], ["known", "<i4"], ["end", "<i4"],
            ["cell_start", "<u4"], ["cell_count", "<u2"], ["source", "u1"], ["flags", "u1"],
            ["peer_type", "u1"], ["pad", "3u1"], ["peer_id", "<i8"], ["group", "<u4"], ["pad2", "<u4"]],
            "flags": {"est": FLAG_EST, "relation": FLAG_RELATION, "derived": FLAG_DERIVED, "extra": FLAG_EXTRA},
            "peer_type": {"none": 0, "company": 1, "person": 2, "address": 3}},
        "cells.bin": {"magic": "TICCEL01", "size": Cell::SIZE, "dtype": [
            ["field", "<u4"], ["code", "<u4"], ["value", "<f4"], ["ctype", "u1"], ["unit", "u1"], ["pad", "<u2"]],
            "ctype": {"num": 0, "cat": 1, "bool": 2, "days": 3, "months": 4}},
        "entities.bin": {"magic": "TICENT01", "size": EntityRow::SIZE, "dtype": [
            ["entity", "<i4"], ["legal_form", "u1"], ["pad", "3u1"], ["registered", "<i4"], ["ceased", "<i4"],
            ["rec_start", "<u8"], ["rec_count", "<u4"], ["pad2", "<u4"]]},
        "economy.bin": {"magic": "TICECO01", "size": EconomyRow::SIZE, "dtype": [
            ["series", "<u4"], ["period", "<i4"], ["published", "<i4"], ["value", "<f4"]]},
        "samples.bin": {"magic": "TICSMP01", "size": SampleRow::SIZE, "dtype": [
            ["entity", "<i4"], ["as_of", "<i4"], ["split", "u1"], ["flags", "u1"], ["query", "u1"], ["horizon", "u1"],
            ["weight", "<f4"],
            ["labels", [["event", "u1"], ["censored", "u1"], ["pad", "<u2"], ["days", "<i4"]], 4],
            ["row", "<u4"], ["pad2", "12u1"]],
            "horizons": HORIZONS, "flags": {"positive": SAMPLE_POSITIVE, "competing": SAMPLE_COMPETING, "censored": SAMPLE_CENSORED},
            "events": {"none": 0, "bankruptcy": 1, "compulsory_liquidation": 2, "reconstruction": 3, "deregistration": 4, "incomplete": EVENT_INCOMPLETE}},
        "index.bin": {"magic": "TICIDX01", "size": IndexRow::SIZE, "dtype": [
            ["tok_start", "<u8"], ["tok_count", "<u4"], ["target_count", "<u4"], ["slot_start", "<u8"], ["slot_count", "<u4"],
            ["extra_start", "<u4"], ["extra_count", "<u4"], ["flags", "<u4"], ["pad", "<u8"]],
            "flags": {"focal_truncated": INDEX_FOCAL_TRUNCATED, "neighbours_truncated": INDEX_NEIGHBOURS_TRUNCATED, "bulk_address": INDEX_BULK_ADDRESS}},
        "tokens.bin": {"magic": "TICTOK01", "size": SeqTok::SIZE, "dtype": [
            ["record", "<u4"], ["slot", "<u2"], ["hop", "u1"], ["path", "u1"]], "extra_bit": EXTRA_BIT},
        "slots.bin": {"magic": "TICSLT01", "size": SlotRow::SIZE, "dtype": [
            ["slot", "<u2"], ["hop", "u1"], ["path", "u1"], ["entity", "<i4"], ["via", "<u4"], ["pad", "<u4"]], "no_via": NO_VIA},
        "extra.bin": {"same_as": "records.bin"},
        "extra_cells.bin": {"same_as": "cells.bin"}
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn roundtrip<T: Rec + PartialEq + std::fmt::Debug>(v: T) {
        let mut b = vec![0u8; T::SIZE];
        v.write(&mut b);
        assert_eq!(T::read(&b), v);
    }

    #[test]
    fn layouts_round_trip() {
        roundtrip(Record {
            entity: -5,
            kind: 7,
            valid: 20000,
            known: 20001,
            end: DAY_MAX,
            cell_start: 123_456,
            cell_count: 61,
            source: 2,
            flags: FLAG_RELATION | FLAG_EST,
            peer_type: PeerType::Address,
            peer_id: -1,
            group: 99,
        });
        roundtrip(Cell {
            field: 3,
            code: 4,
            value: -1.5,
            ctype: CellType::Days,
            unit: 5,
        });
        roundtrip(EntityRow {
            entity: 1,
            legal_form: 0,
            registered: -3000,
            ceased: DAY_MAX,
            rec_start: 1 << 40,
            rec_count: 12,
        });
        roundtrip(EconomyRow {
            series: 1,
            period: 2,
            published: 3,
            value: 4.5,
        });
        roundtrip(SampleRow {
            entity: 9,
            as_of: 20100,
            split: 1,
            flags: SAMPLE_POSITIVE,
            query: 0,
            horizon: 12,
            weight: 20.0,
            labels: [
                HorizonLabel {
                    event: 1,
                    censored: 0,
                    days: 40,
                },
                HorizonLabel::default(),
                HorizonLabel {
                    event: EVENT_INCOMPLETE,
                    censored: 0,
                    days: -1,
                },
                HorizonLabel::default(),
            ],
            row: 77,
        });
        roundtrip(IndexRow {
            tok_start: 1,
            tok_count: 2,
            target_count: 3,
            slot_start: 4,
            slot_count: 5,
            extra_start: 6,
            extra_count: 7,
            flags: 8,
        });
        roundtrip(SeqTok {
            record: EXTRA_BIT | 5,
            slot: 3,
            hop: 2,
            path: 1,
        });
        roundtrip(SlotRow {
            slot: 1,
            hop: 1,
            path: 2,
            entity: 55,
            via: 9,
        });
    }

    #[test]
    fn files_round_trip_with_header_checks() {
        let dir = std::env::temp_dir().join(format!("risk-tokens-rec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cells.bin");
        let cells: Vec<Cell> = (0..70_000)
            .map(|i| Cell {
                field: i,
                code: 0,
                value: i as f32,
                ctype: CellType::Num,
                unit: 1,
            })
            .collect();
        write_all(&path, CELLS_MAGIC, &cells).unwrap();
        let back: Vec<Cell> = read_all(&path, CELLS_MAGIC).unwrap();
        assert_eq!(back, cells);
        assert!(
            read_all::<Cell>(&path, RECORDS_MAGIC).is_err(),
            "magic mismatch"
        );
        assert!(
            read_all::<Record>(&path, CELLS_MAGIC).is_err(),
            "size mismatch"
        );
        let r = Record {
            end: 100,
            known: 120,
            ..Default::default()
        };
        assert_eq!(r.end_known(), 120);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
