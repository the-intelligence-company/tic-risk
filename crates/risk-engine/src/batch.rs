//! a packed batch: the arrays the forward pass reads (int32 ids, float values and masks) plus what
//! it derives from them: the token segments of the recurrence, the readout masks, each sample's
//! token range, the economy row of its as-of day. two sources, one layout: a packed shard file
//! and one sample packed in memory (`risk_model::live`).

use std::path::Path;

use risk_model::live::PackedSample;
use risk_model::shard::Shard;

use crate::artefacts::{Economy, Vocab};
use crate::error::Result;

pub const F_EXTRA: i32 = 8;
pub const CT_CAT: i32 = 1;

/// the next shape bucket above `n`: from `start`, steps of `num/den`, rounded up to a multiple
/// of 32, always strictly above `n`. a sample is padded to a bucket so its shapes repeat across
/// companies; the pad is a quarter at most for tokens (`bucket(n, 64, 5, 4)`) and a half for
/// cells (`bucket(c, 256, 3, 2)`).
pub fn bucket(n: usize, start: usize, num: usize, den: usize) -> usize {
    let mut b = start.div_ceil(32) * 32;
    while b <= n {
        b = (b * num / den).div_ceil(32) * 32;
    }
    b
}

/// the arrays before derivation, a shard's columns widened to i32 and f32
pub struct RawArrays {
    pub path: String,
    pub rows: usize,
    pub seq_len: usize,
    pub samples: usize,
    pub tok_kind: Vec<i32>,
    pub tok_source: Vec<i32>,
    pub tok_flags: Vec<i32>,
    pub tok_hop: Vec<i32>,
    pub tok_path: Vec<i32>,
    pub tok_slot: Vec<i32>,
    pub tok_group: Vec<i32>,
    pub tok_sample: Vec<i32>,
    pub tok_dt: Vec<i32>,
    pub tok_lag: Vec<i32>,
    pub tok_ended: Vec<i32>,
    pub tok_month: Vec<i32>,
    pub tok_cell_start: Vec<i32>,
    pub tok_cell_count: Vec<i32>,
    pub cell_field: Vec<i32>,
    pub cell_code: Vec<i32>,
    pub cell_value: Vec<f32>,
    pub cell_ctype: Vec<i32>,
    pub cell_unit: Vec<i32>,
    pub cell_tok: Vec<i32>,
    pub smp_anchor: Vec<u32>,
    pub smp_entity: Vec<i32>,
    pub smp_as_of: Vec<i32>,
    pub smp_query: Vec<u8>,
    pub smp_horizon: Vec<u8>,
    pub smp_label_event: Vec<u8>,
    pub smp_split: Vec<u8>,
    pub smp_row: Vec<u16>,
    pub smp_tok_start: Vec<u16>,
    pub smp_tok_count: Vec<u16>,
}

pub struct HostBatch {
    pub path: String,
    pub rows: usize,
    pub seq_len: usize,
    pub n_tok: usize,
    pub samples: usize,
    pub cells: usize,
    /// the most tokens any sample spans
    pub max_sample_tokens: usize,
    pub tok_kind: Vec<i32>,
    pub tok_source: Vec<i32>,
    pub tok_flags: Vec<i32>,
    pub tok_hop: Vec<i32>,
    pub tok_path: Vec<i32>,
    pub tok_slot: Vec<i32>,
    pub tok_group: Vec<i32>,
    pub tok_sample: Vec<i32>,
    pub tok_dt: Vec<i32>,
    pub tok_lag: Vec<i32>,
    pub tok_ended: Vec<i32>,
    pub tok_month: Vec<i32>,
    pub tok_cell_start: Vec<i32>,
    pub tok_cell_count: Vec<i32>,
    pub cell_field: Vec<i32>,
    pub cell_code: Vec<i32>,
    pub cell_value: Vec<f32>,
    pub cell_ctype: Vec<i32>,
    pub cell_unit: Vec<i32>,
    pub cell_tok: Vec<i32>,
    pub smp_anchor: Vec<u32>,
    pub smp_entity: Vec<i32>,
    pub smp_as_of: Vec<i32>,
    pub smp_query: Vec<u8>,
    pub smp_horizon: Vec<u8>,
    pub smp_label_event: Vec<u8>,
    pub smp_split: Vec<u8>,
    // derived
    pub valid: Vec<f32>,
    /// token offsets of the recurrence's segments, [n_seg + 1]
    pub segments: Vec<i32>,
    pub seg_id: Vec<i32>,
    pub is_query: Vec<f32>,
    pub is_state: Vec<f32>,
    pub is_focal: Vec<f32>,
    /// the token's day relative to T (−dt), what the rotary time embedding reads
    pub days: Vec<f32>,
    pub sample_clamped: Vec<i32>,
    pub smp_tok_begin: Vec<i32>,
    pub smp_tok_end: Vec<i32>,
    pub has_code: Vec<f32>,
    pub is_numlike: Vec<f32>,
    pub unit_c: Vec<i32>,
    pub source_c: Vec<i32>,
    pub hop_c: Vec<i32>,
    pub eco_row: Vec<i32>,
}

fn i32s<T: Copy + Into<i64>>(v: &[T]) -> Vec<i32> {
    v.iter().map(|&x| x.into() as i32).collect()
}

impl RawArrays {
    pub fn from_shard(sh: &Shard, path: String) -> Result<RawArrays> {
        let h = &sh.header;
        Ok(RawArrays {
            path,
            rows: h.rows,
            seq_len: h.seq_len,
            samples: h.samples,
            tok_kind: i32s(&sh.u16("tok_kind")?),
            tok_source: i32s(&sh.u8("tok_source")?),
            tok_flags: i32s(&sh.u8("tok_flags")?),
            tok_hop: i32s(&sh.u8("tok_hop")?),
            tok_path: i32s(&sh.u16("tok_path")?),
            tok_slot: i32s(&sh.u16("tok_slot")?),
            tok_group: i32s(&sh.u16("tok_group")?),
            tok_sample: i32s(&sh.i16("tok_sample")?),
            tok_dt: sh.i32("tok_dt")?,
            tok_lag: sh.i32("tok_lag")?,
            tok_ended: sh.i32("tok_ended")?,
            tok_month: i32s(&sh.u8("tok_month")?),
            tok_cell_start: i32s(&sh.u32("tok_cell_start")?),
            tok_cell_count: i32s(&sh.u8("tok_cell_count")?),
            cell_field: i32s(&sh.u16("cell_field")?),
            cell_code: i32s(&sh.u16("cell_code")?),
            cell_value: sh.f32("cell_value")?,
            cell_ctype: i32s(&sh.u8("cell_ctype")?),
            cell_unit: i32s(&sh.u8("cell_unit")?),
            cell_tok: i32s(&sh.u32("cell_tok")?),
            smp_anchor: sh.u32("smp_anchor")?,
            smp_entity: sh.i32("smp_entity")?,
            smp_as_of: sh.i32("smp_as_of")?,
            smp_query: sh.u8("smp_query")?,
            smp_horizon: sh.u8("smp_horizon")?,
            smp_label_event: sh.u8("smp_label_event")?,
            smp_split: sh.u8("smp_split")?,
            smp_row: sh.u16("smp_row")?,
            smp_tok_start: sh.u16("smp_tok_start")?,
            smp_tok_count: sh.u16("smp_tok_count")?,
        })
    }

    /// one live sample as a one-row batch, padded to the shape buckets: pad tokens (sample −1,
    /// the pad kind, no cells) after the real ones, so they form their own recurrence segment,
    /// are masked from every real token's attention and from the readout, and are zeroed by the
    /// valid mask; pad cells belong to the first pad token and no token pools them
    pub fn from_packed(p: &PackedSample) -> RawArrays {
        let n = p.n_tok;
        let c = p.cell_field.len();
        let (n_pad, c_pad) = (bucket(n, 64, 5, 4), bucket(c, 256, 3, 2));
        fn pad<T: Copy>(v: &[T], len: usize, fill: T) -> Vec<T> {
            let mut out = Vec::with_capacity(len);
            out.extend_from_slice(v);
            out.resize(len, fill);
            out
        }
        fn pad_i<T: Copy + Into<i64>>(v: &[T], len: usize, fill: i32) -> Vec<i32> {
            let mut out = i32s(v);
            out.resize(len, fill);
            out
        }
        RawArrays {
            path: format!("live entity {} as of {}", p.entity, p.as_of),
            rows: 1,
            seq_len: n_pad,
            samples: 1,
            tok_kind: pad_i(&p.tok_kind, n_pad, 0),
            tok_source: pad_i(&p.tok_source, n_pad, 0),
            tok_flags: pad_i(&p.tok_flags, n_pad, 0),
            tok_hop: pad_i(&p.tok_hop, n_pad, 0),
            tok_path: pad_i(&p.tok_path, n_pad, 0),
            tok_slot: pad_i(&p.tok_slot, n_pad, 0),
            tok_group: pad_i(&p.tok_group, n_pad, 0),
            tok_sample: { let mut v = vec![0; n]; v.resize(n_pad, -1); v },
            tok_dt: pad(&p.tok_dt, n_pad, 0),
            tok_lag: pad(&p.tok_lag, n_pad, 0),
            tok_ended: pad(&p.tok_ended, n_pad, -1),
            tok_month: pad_i(&p.tok_month, n_pad, 0),
            tok_cell_start: pad_i(&p.tok_cell_start, n_pad, 0),
            tok_cell_count: pad_i(&p.tok_cell_count, n_pad, 0),
            cell_field: pad_i(&p.cell_field, c_pad, 0),
            cell_code: pad_i(&p.cell_code, c_pad, 0),
            cell_value: pad(&p.cell_value, c_pad, 0.0),
            cell_ctype: pad_i(&p.cell_ctype, c_pad, 0),
            cell_unit: pad_i(&p.cell_unit, c_pad, 0),
            cell_tok: pad_i(&p.cell_tok, c_pad, n as i32),
            smp_anchor: vec![0],
            smp_entity: vec![p.entity],
            smp_as_of: vec![p.as_of],
            smp_query: vec![p.query],
            smp_horizon: vec![p.horizon],
            smp_label_event: vec![255; 4],
            smp_split: vec![1],
            smp_row: vec![0],
            smp_tok_start: vec![0],
            smp_tok_count: vec![p.n_tok as u16],
        }
    }
}

impl HostBatch {
    pub fn read(path: &Path, vocab: &Vocab, economy: Option<&Economy>) -> Result<HostBatch> {
        let sh = Shard::read(path)?;
        let raw = RawArrays::from_shard(&sh, path.display().to_string())?;
        Ok(Self::from_raw(raw, vocab, economy))
    }

    pub fn from_packed(p: &PackedSample, vocab: &Vocab, economy: Option<&Economy>) -> HostBatch {
        Self::from_raw(RawArrays::from_packed(p), vocab, economy)
    }

    pub fn from_raw(r: RawArrays, vocab: &Vocab, economy: Option<&Economy>) -> HostBatch {
        let (rows, seq_len, samples) = (r.rows, r.seq_len, r.samples);
        let n_tok = rows * seq_len;
        let cells = r.cell_field.len();
        let valid: Vec<f32> = r.tok_sample.iter().map(|&s| if s >= 0 { 1.0 } else { 0.0 }).collect();
        // segments: a run of one sample within a row; a row boundary always starts one (cu_seqlens_of)
        let mut segments = Vec::with_capacity(samples + rows + 1);
        let mut seg_id = vec![0i32; n_tok];
        for (t, sid) in seg_id.iter_mut().enumerate() {
            if t % seq_len == 0 || r.tok_sample[t] != r.tok_sample[t - 1] {
                segments.push(t as i32);
            }
            *sid = segments.len() as i32 - 1;
        }
        segments.push(n_tok as i32);
        let is_query: Vec<f32> = (0..n_tok).map(|t| if r.tok_kind[t] == vocab.query_kind && valid[t] > 0.0 { 1.0 } else { 0.0 }).collect();
        let is_state: Vec<f32> = (0..n_tok)
            .map(|t| if vocab.is_state_kind.get(r.tok_kind[t] as usize).copied().unwrap_or(0) == 1 && valid[t] > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let is_focal: Vec<f32> = (0..n_tok).map(|t| if r.tok_slot[t] == 0 && (r.tok_flags[t] & F_EXTRA) == 0 && valid[t] > 0.0 { 1.0 } else { 0.0 }).collect();
        let days: Vec<f32> = r.tok_dt.iter().map(|&d| -(d as f32)).collect();
        let sample_clamped: Vec<i32> = r.tok_sample.iter().map(|&s| s.max(0)).collect();
        let smp_tok_begin: Vec<i32> = (0..samples).map(|s| (r.smp_row[s] as usize * seq_len + r.smp_tok_start[s] as usize) as i32).collect();
        let smp_tok_end: Vec<i32> = (0..samples).map(|s| smp_tok_begin[s] + r.smp_tok_count[s] as i32).collect();
        let max_sample_tokens = r.smp_tok_count.iter().map(|&c| c as usize).max().unwrap_or(0);
        let has_code: Vec<f32> = (0..cells).map(|c| if r.cell_code[c] > 0 || r.cell_ctype[c] == CT_CAT { 1.0 } else { 0.0 }).collect();
        let is_numlike: Vec<f32> = r.cell_ctype.iter().map(|&t| if t != CT_CAT { 1.0 } else { 0.0 }).collect();
        let unit_c: Vec<i32> = r.cell_unit.iter().map(|&u| u.min(15)).collect();
        let source_c: Vec<i32> = r.tok_source.iter().map(|&u| u.min(15)).collect();
        let hop_c: Vec<i32> = r.tok_hop.iter().map(|&u| u.min(7)).collect();
        let eco_row: Vec<i32> = match economy {
            Some(e) => r.smp_as_of.iter().map(|&d| e.row_of_day(d) as i32).collect(),
            None => vec![0; samples],
        };
        HostBatch {
            path: r.path, rows, seq_len, n_tok, samples, cells, max_sample_tokens,
            tok_kind: r.tok_kind, tok_source: r.tok_source, tok_flags: r.tok_flags, tok_hop: r.tok_hop, tok_path: r.tok_path,
            tok_slot: r.tok_slot, tok_group: r.tok_group, tok_sample: r.tok_sample, tok_dt: r.tok_dt, tok_lag: r.tok_lag,
            tok_ended: r.tok_ended, tok_month: r.tok_month, tok_cell_start: r.tok_cell_start, tok_cell_count: r.tok_cell_count,
            cell_field: r.cell_field, cell_code: r.cell_code, cell_value: r.cell_value, cell_ctype: r.cell_ctype, cell_unit: r.cell_unit, cell_tok: r.cell_tok,
            smp_anchor: r.smp_anchor, smp_entity: r.smp_entity, smp_as_of: r.smp_as_of, smp_query: r.smp_query, smp_horizon: r.smp_horizon,
            smp_label_event: r.smp_label_event, smp_split: r.smp_split,
            valid, segments, seg_id, is_query, is_state, is_focal, days, sample_clamped, smp_tok_begin, smp_tok_end,
            has_code, is_numlike, unit_c, source_c, hop_c, eco_row,
        }
    }

    pub fn n_segments(&self) -> usize {
        self.segments.len() - 1
    }
}
