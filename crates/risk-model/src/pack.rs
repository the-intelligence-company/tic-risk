//! the packer: resolves a materialised anchor set into model ids and normalised values, orders a
//! sample's tokens chronologically across the neighbourhood with the extra records last, packs
//! samples into rows of `seq_len` tokens and writes shards per split. pairs of anchors of one
//! entity stay in one shard for the contrastive objective.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Serialize;

use crate::Error;
use crate::dates::{Day, to_naive};
use crate::shard::{Column, Header, VERSION, write};
use crate::vocab::ModelDir;
use risk_tokens::record::{
    CELLS_MAGIC, Cell, CellType, EXTRA_BIT, FLAG_DERIVED, FLAG_EST, FLAG_EXTRA, FLAG_RELATION,
    INDEX_MAGIC, IndexRow, RECORDS_MAGIC, Record, SampleRow, SeqTok, TOKENS_MAGIC, read_all,
};
use risk_tokens::streams::Streams;

pub const FLAG_ENDED: u8 = 16;

#[derive(Debug, Clone)]
pub struct Options {
    pub seq_len: usize,
    pub rows_per_shard: usize,
    pub seed: u64,
    pub pairs: bool,
    pub corpus: String,
    pub corpus_index: u16,
    pub max_cells_per_token: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            seq_len: 2048,
            rows_per_shard: 64,
            seed: 1,
            pairs: false,
            corpus: "corpus".into(),
            corpus_index: 0,
            max_cells_per_token: 255,
        }
    }
}

#[derive(Debug, Default, Serialize, Clone)]
pub struct PackStats {
    pub split: String,
    pub shards: usize,
    pub samples: u64,
    pub rows: u64,
    pub tokens: u64,
    pub cells: u64,
    pub targets: u64,
    pub truncated_samples: u64,
    pub pairs: u64,
    pub unknown_ids: u64,
    pub fill: f64,
    pub seconds: f64,
}

/// one resolved sample
pub(crate) struct Resolved {
    pub(crate) sample: SampleRow,
    pub(crate) index_flags: u32,
    pub(crate) toks: Vec<Tok>,
    pub(crate) cells: Vec<PCell>,
    pub(crate) targets: Vec<Tgt>,
    pub(crate) tcells: Vec<PCell>,
    pub(crate) truncated: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct Tok {
    pub(crate) kind: u16,
    pub(crate) source: u8,
    pub(crate) flags: u8,
    pub(crate) hop: u8,
    pub(crate) path: u16,
    pub(crate) slot: u16,
    pub(crate) group: u16,
    pub(crate) dt: i32,
    pub(crate) lag: i32,
    pub(crate) ended: i32,
    pub(crate) month: u8,
    pub(crate) cell_start: u32,
    pub(crate) cell_count: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct PCell {
    pub(crate) field: u16,
    pub(crate) code: u16,
    pub(crate) value: f32,
    pub(crate) raw: f32,
    pub(crate) ctype: u8,
    pub(crate) unit: u8,
}

#[derive(Clone, Copy)]
pub(crate) struct Tgt {
    kind: u16,
    days: i32,
    cell_start: u32,
    cell_count: u8,
}

pub(crate) struct Materialised {
    pub(crate) samples: Vec<SampleRow>,
    pub(crate) index: Vec<IndexRow>,
    pub(crate) tokens: Vec<SeqTok>,
    pub(crate) extra: Vec<Record>,
    pub(crate) extra_cells: Vec<Cell>,
}

fn load(anchors_dir: &Path) -> Result<Materialised, Error> {
    use risk_tokens::record::SAMPLES_MAGIC;
    Ok(Materialised {
        samples: read_all(&anchors_dir.join("samples.bin"), SAMPLES_MAGIC)?,
        index: read_all(&anchors_dir.join("index.bin"), INDEX_MAGIC)?,
        tokens: read_all(&anchors_dir.join("tokens.bin"), TOKENS_MAGIC)?,
        extra: read_all(&anchors_dir.join("extra.bin"), RECORDS_MAGIC)?,
        extra_cells: read_all(&anchors_dir.join("extra_cells.bin"), CELLS_MAGIC)?,
    })
}

pub(crate) struct Ctx<'a> {
    pub(crate) s: &'a Streams,
    pub(crate) m: &'a Materialised,
    pub(crate) md: &'a ModelDir,
    pub(crate) remap: &'a [u32],
    pub(crate) opts: &'a Options,
}

fn id16(remap: &[u32], id: u32, unknown: &mut u64) -> u16 {
    let v = remap.get(id as usize).copied().unwrap_or(1);
    if v == 1 && id != 1 {
        *unknown += 1;
    }
    u16::try_from(v).unwrap_or(1)
}

fn norm(md: &ModelDir, field: u16, ctype: CellType, raw: f32) -> f32 {
    match ctype {
        CellType::Cat => 0.0,
        CellType::Bool => raw,
        _ => {
            let (scale, mean, std) = md
                .norm
                .get(field as usize)
                .copied()
                .unwrap_or((1.0, 0.0, 1.0));
            let v = ((raw / scale) as f64).asinh() as f32;
            ((v - mean) / std).clamp(-20.0, 20.0)
        }
    }
}

fn cells_of<'a>(ctx: &Ctx<'a>, r: &Record, extra: bool) -> &'a [Cell] {
    let src = if extra {
        &ctx.m.extra_cells
    } else {
        &ctx.s.cells
    };
    let range = r.cells();
    src.get(range).unwrap_or(&[])
}

pub(crate) fn resolve(ctx: &Ctx, i: usize, unknown: &mut u64) -> Resolved {
    let sample = ctx.m.samples[i];
    let ix = ctx.m.index[i];
    let t: Day = sample.as_of;
    let tok_range = ix.tok_start as usize..(ix.tok_start + u64::from(ix.tok_count)) as usize;
    let tgt_range = (ix.tok_start + u64::from(ix.tok_count)) as usize
        ..(ix.tok_start + u64::from(ix.tok_count) + u64::from(ix.target_count)) as usize;
    // (valid, emission order) → chronological with the extras last in emission order
    let mut order: Vec<(Day, usize)> = ctx.m.tokens[tok_range.clone()]
        .iter()
        .enumerate()
        .map(|(k, st)| {
            let r = record_of(ctx, st, &ix);
            (r.valid, k)
        })
        .collect();
    order.sort_unstable();
    let mut truncated = false;
    if order.len() > ctx.opts.seq_len {
        let drop = order.len() - ctx.opts.seq_len;
        // drop the oldest non-extra tokens
        let mut kept = Vec::with_capacity(ctx.opts.seq_len);
        let mut dropped = 0;
        for (v, k) in order {
            let st = &ctx.m.tokens[tok_range.start + k];
            let is_extra = st.record & EXTRA_BIT != 0;
            if dropped < drop && !is_extra {
                dropped += 1;
                continue;
            }
            kept.push((v, k));
        }
        order = kept;
        truncated = true;
    }
    let mut groups: HashMap<u32, u16> = HashMap::new();
    let mut toks = Vec::with_capacity(order.len());
    let mut cells = Vec::with_capacity(order.len() * 4);
    for (_, k) in order {
        let st = &ctx.m.tokens[tok_range.start + k];
        let is_extra = st.record & EXTRA_BIT != 0;
        let r = record_of(ctx, st, &ix);
        let ended = r.is_relation() && r.end <= t;
        let mut flags = 0u8;
        if r.flags & FLAG_EST != 0 {
            flags |= 1;
        }
        if r.flags & FLAG_RELATION != 0 {
            flags |= 2;
        }
        if r.flags & FLAG_DERIVED != 0 {
            flags |= 4;
        }
        if is_extra || r.flags & FLAG_EXTRA != 0 {
            flags |= 8;
        }
        if ended {
            flags |= FLAG_ENDED;
        }
        let group = if r.group == 0 {
            0
        } else {
            let n = groups.len() as u16 + 1;
            *groups.entry(r.group).or_insert(n)
        };
        let cell_start = cells.len() as u32;
        let rc = cells_of(ctx, r, is_extra);
        let n = rc.len().min(ctx.opts.max_cells_per_token);
        for c in &rc[..n] {
            let field = id16(ctx.remap, c.field, unknown);
            let code = if c.code == 0 {
                0
            } else {
                id16(ctx.remap, c.code, unknown)
            };
            cells.push(PCell {
                field,
                code,
                value: norm(ctx.md, field, c.ctype, c.value),
                raw: c.value,
                ctype: c.ctype as u8,
                unit: c.unit,
            });
        }
        use chrono::Datelike;
        toks.push(Tok {
            kind: id16(ctx.remap, r.kind, unknown),
            source: r.source,
            flags,
            hop: st.hop,
            path: id16(ctx.remap, u32::from(st.path), unknown),
            slot: st.slot,
            group,
            dt: t - r.valid,
            lag: r.known - r.valid,
            ended: if ended { t - r.end } else { -1 },
            month: to_naive(r.valid).month() as u8,
            cell_start,
            cell_count: n as u8,
        });
    }
    let mut targets = Vec::with_capacity(ix.target_count as usize);
    let mut tcells = Vec::new();
    for st in &ctx.m.tokens[tgt_range] {
        let r = &ctx.s.records[st.record as usize];
        let cell_start = tcells.len() as u32;
        let rc = cells_of(ctx, r, false);
        let n = rc.len().min(ctx.opts.max_cells_per_token);
        for c in &rc[..n] {
            let field = id16(ctx.remap, c.field, unknown);
            let code = if c.code == 0 {
                0
            } else {
                id16(ctx.remap, c.code, unknown)
            };
            tcells.push(PCell {
                field,
                code,
                value: norm(ctx.md, field, c.ctype, c.value),
                raw: c.value,
                ctype: c.ctype as u8,
                unit: c.unit,
            });
        }
        targets.push(Tgt {
            kind: id16(ctx.remap, r.kind, unknown),
            days: r.valid - t,
            cell_start,
            cell_count: n as u8,
        });
    }
    Resolved {
        sample,
        index_flags: ix.flags,
        toks,
        cells,
        targets,
        tcells,
        truncated,
    }
}

fn record_of<'a>(ctx: &Ctx<'a>, st: &SeqTok, ix: &IndexRow) -> &'a Record {
    if st.record & EXTRA_BIT != 0 {
        let k = (st.record & (EXTRA_BIT - 1)) as usize;
        &ctx.m.extra[ix.extra_start as usize + k]
    } else {
        &ctx.s.records[st.record as usize]
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// the sample order of one split: shuffled by seed; with pairs, two anchors of an entity adjacent
fn order_split(
    samples: &[SampleRow],
    split: u8,
    seed: u64,
    pairs: bool,
) -> (Vec<(usize, Option<usize>)>, u64) {
    let idx: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter(|(_, r)| r.split == split)
        .map(|(i, _)| i)
        .collect();
    if !pairs {
        let mut keyed: Vec<(u64, usize)> = idx
            .iter()
            .map(|&i| (splitmix(seed ^ (i as u64) << 1), i))
            .collect();
        keyed.sort_unstable();
        return (keyed.into_iter().map(|(_, i)| (i, None)).collect(), 0);
    }
    let mut by_entity: HashMap<i32, Vec<usize>> = HashMap::new();
    for &i in &idx {
        by_entity.entry(samples[i].entity).or_default().push(i);
    }
    let mut units: Vec<(u64, usize, Option<usize>)> = Vec::with_capacity(idx.len());
    let mut n_pairs = 0u64;
    for (e, mut list) in by_entity {
        list.sort_by_key(|&i| (splitmix(seed ^ (i as u64) << 3), i));
        let mut it = list.into_iter();
        while let Some(a) = it.next() {
            let key = splitmix(seed ^ (e as u64) << 7 ^ a as u64);
            match it.next() {
                Some(b) => {
                    units.push((key, a, Some(b)));
                    n_pairs += 1;
                }
                None => units.push((key, a, None)),
            }
        }
    }
    units.sort_unstable();
    (units.into_iter().map(|(_, a, b)| (a, b)).collect(), n_pairs)
}

struct ShardBuf {
    rows: Vec<Vec<usize>>, // sample positions per row
    lens: Vec<usize>,
}

impl ShardBuf {
    fn new() -> ShardBuf {
        ShardBuf {
            rows: Vec::new(),
            lens: Vec::new(),
        }
    }

    /// whether the samples with these lengths can all be placed, in order, by the first-fit rule
    fn can_place(&self, lens: &[usize], seq_len: usize, max_rows: usize) -> bool {
        let mut used = self.lens.clone();
        for &len in lens {
            if len > seq_len {
                return false;
            }
            match used.iter_mut().find(|u| **u + len <= seq_len) {
                Some(u) => *u += len,
                None => {
                    if used.len() >= max_rows {
                        return false;
                    }
                    used.push(len);
                }
            }
        }
        true
    }

    /// first row with room, else a new row; returns (row, start)
    fn place(&mut self, len: usize, seq_len: usize, max_rows: usize) -> Option<(usize, usize)> {
        for (r, used) in self.lens.iter_mut().enumerate() {
            if *used + len <= seq_len {
                let start = *used;
                *used += len;
                self.rows[r].push(start);
                return Some((r, start));
            }
        }
        if self.rows.len() < max_rows {
            self.rows.push(vec![0]);
            self.lens.push(len);
            return Some((self.rows.len() - 1, 0));
        }
        None
    }
}

#[allow(clippy::too_many_lines)]
fn write_shard(
    ctx: &Ctx,
    out: &Path,
    shard_no: usize,
    split: &str,
    placed: &[(Resolved, usize, usize, i32)],
    rows: usize,
    anchors_name: &str,
) -> Result<(u64, u64, u64, u64), Error> {
    let l = ctx.opts.seq_len;
    let n = rows * l;
    let mut tok_kind = vec![0u16; n];
    let mut tok_source = vec![0u8; n];
    let mut tok_flags = vec![0u8; n];
    let mut tok_hop = vec![0u8; n];
    let mut tok_path = vec![0u16; n];
    let mut tok_slot = vec![0u16; n];
    let mut tok_group = vec![0u16; n];
    let mut tok_sample = vec![-1i16; n];
    let mut tok_dt = vec![0i32; n];
    let mut tok_lag = vec![0i32; n];
    let mut tok_ended = vec![-1i32; n];
    let mut tok_month = vec![0u8; n];
    let mut tok_cell_start = vec![0u32; n];
    let mut tok_cell_count = vec![0u8; n];
    let mut cell_field = Vec::new();
    let mut cell_code = Vec::new();
    let mut cell_value = Vec::new();
    let mut cell_raw = Vec::new();
    let mut cell_ctype = Vec::new();
    let mut cell_unit = Vec::new();
    let mut cell_tok = Vec::new();
    let s_n = placed.len();
    let mut smp_row = vec![0u16; s_n];
    let mut smp_tok_start = vec![0u16; s_n];
    let mut smp_tok_count = vec![0u16; s_n];
    let mut smp_entity = vec![0i32; s_n];
    let mut smp_as_of = vec![0i32; s_n];
    let mut smp_query = vec![0u8; s_n];
    let mut smp_horizon = vec![0u8; s_n];
    let mut smp_split = vec![0u8; s_n];
    let mut smp_flags = vec![0u8; s_n];
    let mut smp_weight = vec![0f32; s_n];
    let mut smp_anchor = vec![0u32; s_n];
    let mut smp_corpus = vec![ctx.opts.corpus_index; s_n];
    let mut smp_label_event = vec![0u8; s_n * 4];
    let mut smp_label_censored = vec![0u8; s_n * 4];
    let mut smp_label_days = vec![0i32; s_n * 4];
    let mut smp_pair = vec![-1i32; s_n];
    let mut smp_tgt_start = vec![0u32; s_n];
    let mut smp_tgt_count = vec![0u16; s_n];
    let mut smp_index_flags = vec![0u32; s_n];
    let mut tgt_sample = Vec::new();
    let mut tgt_kind = Vec::new();
    let mut tgt_days = Vec::new();
    let mut tgt_cell_start = Vec::new();
    let mut tgt_cell_count = Vec::new();
    let mut tcell_field = Vec::new();
    let mut tcell_code = Vec::new();
    let mut tcell_value = Vec::new();
    let mut tcell_raw = Vec::new();
    let mut tcell_ctype = Vec::new();
    let mut tcell_unit = Vec::new();
    let mut tokens = 0u64;
    for (si, (r, row, start, pair)) in placed.iter().enumerate() {
        smp_row[si] = *row as u16;
        smp_tok_start[si] = *start as u16;
        smp_tok_count[si] = r.toks.len() as u16;
        smp_entity[si] = r.sample.entity;
        smp_as_of[si] = r.sample.as_of;
        smp_query[si] = r.sample.query;
        smp_horizon[si] = r.sample.horizon;
        smp_split[si] = r.sample.split;
        smp_flags[si] = r.sample.flags;
        smp_weight[si] = r.sample.weight;
        smp_anchor[si] = r.sample.row;
        smp_corpus[si] = ctx.opts.corpus_index;
        for (h, lab) in r.sample.labels.iter().enumerate() {
            smp_label_event[si * 4 + h] = lab.event;
            smp_label_censored[si * 4 + h] = lab.censored;
            smp_label_days[si * 4 + h] = lab.days;
        }
        smp_pair[si] = *pair;
        smp_index_flags[si] = r.index_flags;
        let base = row * l + start;
        let cell_base = cell_field.len() as u32;
        for (k, t) in r.toks.iter().enumerate() {
            let p = base + k;
            tok_kind[p] = t.kind;
            tok_source[p] = t.source;
            tok_flags[p] = t.flags;
            tok_hop[p] = t.hop;
            tok_path[p] = t.path;
            tok_slot[p] = t.slot;
            tok_group[p] = t.group;
            tok_sample[p] = si as i16;
            tok_dt[p] = t.dt;
            tok_lag[p] = t.lag;
            tok_ended[p] = t.ended;
            tok_month[p] = t.month;
            tok_cell_start[p] = cell_base + t.cell_start;
            tok_cell_count[p] = t.cell_count;
            for c in
                &r.cells[t.cell_start as usize..t.cell_start as usize + usize::from(t.cell_count)]
            {
                cell_field.push(c.field);
                cell_code.push(c.code);
                cell_value.push(c.value);
                cell_raw.push(c.raw);
                cell_ctype.push(c.ctype);
                cell_unit.push(c.unit);
                cell_tok.push(p as u32);
            }
        }
        tokens += r.toks.len() as u64;
        smp_tgt_start[si] = tgt_sample.len() as u32;
        smp_tgt_count[si] = r.targets.len() as u16;
        let tc_base = tcell_field.len() as u32;
        for t in &r.targets {
            tgt_sample.push(si as u32);
            tgt_kind.push(t.kind);
            tgt_days.push(t.days);
            tgt_cell_start.push(tc_base + t.cell_start);
            tgt_cell_count.push(t.cell_count);
            for c in
                &r.tcells[t.cell_start as usize..t.cell_start as usize + usize::from(t.cell_count)]
            {
                tcell_field.push(c.field);
                tcell_code.push(c.code);
                tcell_value.push(c.value);
                tcell_raw.push(c.raw);
                tcell_ctype.push(c.ctype);
                tcell_unit.push(c.unit);
            }
        }
    }
    let cells = cell_field.len();
    let targets = tgt_sample.len();
    let tcells = tcell_field.len();
    let header = Header {
        version: VERSION,
        seq_len: l,
        rows,
        samples: s_n,
        cells,
        targets,
        target_cells: tcells,
        vocab_size: ctx.md.vocab.len(),
        vocab_hash: ctx.md.hash.clone(),
        stats_cut: ctx.md.cut.clone(),
        corpora: vec![ctx.opts.corpus.clone()],
        corpus: ctx.opts.corpus_index,
        anchors: anchors_name.to_owned(),
        split: split.to_owned(),
        seed: ctx.opts.seed,
        arrays: vec![],
    };
    let rl = vec![rows, l];
    let s4 = vec![s_n, 4];
    let path = out.join(format!("shard_{shard_no:05}.bin"));
    write(
        &path,
        header,
        vec![
            ("tok_kind", Column::U16(tok_kind), rl.clone()),
            ("tok_source", Column::U8(tok_source), rl.clone()),
            ("tok_flags", Column::U8(tok_flags), rl.clone()),
            ("tok_hop", Column::U8(tok_hop), rl.clone()),
            ("tok_path", Column::U16(tok_path), rl.clone()),
            ("tok_slot", Column::U16(tok_slot), rl.clone()),
            ("tok_group", Column::U16(tok_group), rl.clone()),
            ("tok_sample", Column::I16(tok_sample), rl.clone()),
            ("tok_dt", Column::I32(tok_dt), rl.clone()),
            ("tok_lag", Column::I32(tok_lag), rl.clone()),
            ("tok_ended", Column::I32(tok_ended), rl.clone()),
            ("tok_month", Column::U8(tok_month), rl.clone()),
            ("tok_cell_start", Column::U32(tok_cell_start), rl.clone()),
            ("tok_cell_count", Column::U8(tok_cell_count), rl),
            ("cell_field", Column::U16(cell_field), vec![cells]),
            ("cell_code", Column::U16(cell_code), vec![cells]),
            ("cell_value", Column::F32(cell_value), vec![cells]),
            ("cell_raw", Column::F32(cell_raw), vec![cells]),
            ("cell_ctype", Column::U8(cell_ctype), vec![cells]),
            ("cell_unit", Column::U8(cell_unit), vec![cells]),
            ("cell_tok", Column::U32(cell_tok), vec![cells]),
            ("smp_row", Column::U16(smp_row), vec![s_n]),
            ("smp_tok_start", Column::U16(smp_tok_start), vec![s_n]),
            ("smp_tok_count", Column::U16(smp_tok_count), vec![s_n]),
            ("smp_entity", Column::I32(smp_entity), vec![s_n]),
            ("smp_as_of", Column::I32(smp_as_of), vec![s_n]),
            ("smp_query", Column::U8(smp_query), vec![s_n]),
            ("smp_horizon", Column::U8(smp_horizon), vec![s_n]),
            ("smp_split", Column::U8(smp_split), vec![s_n]),
            ("smp_flags", Column::U8(smp_flags), vec![s_n]),
            ("smp_weight", Column::F32(smp_weight), vec![s_n]),
            ("smp_anchor", Column::U32(smp_anchor), vec![s_n]),
            ("smp_corpus", Column::U16(smp_corpus), vec![s_n]),
            ("smp_label_event", Column::U8(smp_label_event), s4.clone()),
            (
                "smp_label_censored",
                Column::U8(smp_label_censored),
                s4.clone(),
            ),
            ("smp_label_days", Column::I32(smp_label_days), s4),
            ("smp_pair", Column::I32(smp_pair), vec![s_n]),
            ("smp_tgt_start", Column::U32(smp_tgt_start), vec![s_n]),
            ("smp_tgt_count", Column::U16(smp_tgt_count), vec![s_n]),
            ("smp_index_flags", Column::U32(smp_index_flags), vec![s_n]),
            ("tgt_sample", Column::U32(tgt_sample), vec![targets]),
            ("tgt_kind", Column::U16(tgt_kind), vec![targets]),
            ("tgt_days", Column::I32(tgt_days), vec![targets]),
            ("tgt_cell_start", Column::U32(tgt_cell_start), vec![targets]),
            ("tgt_cell_count", Column::U8(tgt_cell_count), vec![targets]),
            ("tcell_field", Column::U16(tcell_field), vec![tcells]),
            ("tcell_code", Column::U16(tcell_code), vec![tcells]),
            ("tcell_value", Column::F32(tcell_value), vec![tcells]),
            ("tcell_raw", Column::F32(tcell_raw), vec![tcells]),
            ("tcell_ctype", Column::U8(tcell_ctype), vec![tcells]),
            ("tcell_unit", Column::U8(tcell_unit), vec![tcells]),
        ],
    )?;
    Ok((tokens, cells as u64, targets as u64, s_n as u64))
}

/// packs the anchor set's splits into `out/<split>/shard_*.bin`
pub fn pack(
    tokens_dir: &Path,
    anchors_name: &str,
    model_dir: &Path,
    out: &Path,
    opts: &Options,
) -> Result<Vec<PackStats>, Error> {
    let t0 = std::time::Instant::now();
    let md = ModelDir::load(model_dir)?;
    let s = Streams::open(tokens_dir)?;
    let (remap, unknown_keys) = md.remap(&s.vocab);
    if unknown_keys > 0 {
        tracing::warn!(
            unknown_keys,
            "corpus keys missing from the model vocabulary map to unknown"
        );
    }
    let anchors_dir = tokens_dir.join("anchors").join(anchors_name);
    let m = load(&anchors_dir)?;
    tracing::info!(
        samples = m.samples.len(),
        tokens = m.tokens.len(),
        "anchor set loaded"
    );
    let ctx = Ctx {
        s: &s,
        m: &m,
        md: &md,
        remap: &remap,
        opts,
    };
    let mut all = Vec::new();
    for (split, name) in [(0u8, "train"), (1u8, "test")] {
        let (order, n_pairs) = order_split(&m.samples, split, opts.seed, opts.pairs);
        if order.is_empty() {
            continue;
        }
        let dir = out.join(name);
        std::fs::create_dir_all(&dir)?;
        let mut st = PackStats {
            split: name.to_owned(),
            pairs: n_pairs,
            ..Default::default()
        };
        let mut shard_no = 0usize;
        let mut buf = ShardBuf::new();
        let mut placed: Vec<(Resolved, usize, usize, i32)> = Vec::new();
        let mut pos = 0usize;
        let chunk = 4096;
        let unknown = 0u64;
        while pos < order.len() {
            let end = (pos + chunk).min(order.len());
            let resolved: Vec<(Resolved, Option<Resolved>)> = order[pos..end]
                .par_iter()
                .map(|(a, b)| {
                    let mut u = 0u64;
                    (
                        resolve(&ctx, *a, &mut u),
                        b.map(|b| resolve(&ctx, b, &mut u)),
                    )
                })
                .collect();
            for (a, b) in resolved {
                let can_place = buf.can_place(
                    &[a.toks.len(), b.as_ref().map_or(0, |b| b.toks.len())]
                        [..1 + usize::from(b.is_some())],
                    opts.seq_len,
                    opts.rows_per_shard,
                );
                if !can_place {
                    // close the shard
                    let rows = buf.rows.len();
                    let (tk, ce, tg, sm) =
                        write_shard(&ctx, &dir, shard_no, name, &placed, rows, anchors_name)?;
                    st.tokens += tk;
                    st.cells += ce;
                    st.targets += tg;
                    st.samples += sm;
                    st.rows += rows as u64;
                    st.shards += 1;
                    shard_no += 1;
                    buf = ShardBuf::new();
                    placed.clear();
                }
                let ia = placed.len() as i32;
                let (ra, sa) = buf
                    .place(a.toks.len(), opts.seq_len, opts.rows_per_shard)
                    .ok_or_else(|| Error::Other("sample does not fit a row".into()))?;
                st.truncated_samples += u64::from(a.truncated);
                let pair_a = if b.is_some() { ia + 1 } else { -1 };
                placed.push((a, ra, sa, pair_a));
                if let Some(b) = b {
                    let (rb, sb) = buf
                        .place(b.toks.len(), opts.seq_len, opts.rows_per_shard)
                        .ok_or_else(|| Error::Other("paired sample does not fit a row".into()))?;
                    st.truncated_samples += u64::from(b.truncated);
                    placed.push((b, rb, sb, ia));
                }
            }
            pos = end;
        }
        if !placed.is_empty() {
            let rows = buf.rows.len();
            let (tk, ce, tg, sm) =
                write_shard(&ctx, &dir, shard_no, name, &placed, rows, anchors_name)?;
            st.tokens += tk;
            st.cells += ce;
            st.targets += tg;
            st.samples += sm;
            st.rows += rows as u64;
            st.shards += 1;
        }
        st.unknown_ids = unknown;
        st.fill = if st.rows > 0 {
            st.tokens as f64 / (st.rows as f64 * opts.seq_len as f64)
        } else {
            0.0
        };
        st.seconds = t0.elapsed().as_secs_f64();
        tracing::info!(
            split = name,
            shards = st.shards,
            samples = st.samples,
            fill = st.fill,
            "packed"
        );
        all.push(st);
    }
    let manifest = serde_json::json!({
        "tokens_dir": tokens_dir.display().to_string(),
        "anchors": anchors_name,
        "model_dir": model_dir.display().to_string(),
        "vocab_hash": md.hash,
        "stats_cut": md.cut,
        "corpus": opts.corpus,
        "corpus_index": opts.corpus_index,
        "seq_len": opts.seq_len,
        "rows_per_shard": opts.rows_per_shard,
        "seed": opts.seed,
        "pairs": opts.pairs,
        "unknown_keys": unknown_keys,
        "splits": all,
    });
    std::fs::write(
        out.join("pack.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(all)
}

pub fn shard_files(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    v.sort();
    Ok(v)
}
