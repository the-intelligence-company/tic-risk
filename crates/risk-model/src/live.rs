//! one live sample packed in memory: the sampler's output for a single (entity, as-of, question)
//! resolved and laid out exactly as one row of a shard, so a live score reads the same arrays, the
//! same id remap and the same normalisation the training batches went through. no file is written.

use risk_tokens::dates::Day;
use risk_tokens::record::{HorizonLabel, IndexRow, SampleRow};
use risk_tokens::sampler::Sample;
use risk_tokens::streams::Streams;

use crate::pack::{Ctx, Materialised, Options, resolve};
use crate::vocab::ModelDir;

/// the arrays of one packed sample, one row of `n_tok` tokens, the shard's column types
#[derive(Debug, Default, Clone)]
pub struct PackedSample {
    pub n_tok: usize,
    pub tok_kind: Vec<u16>,
    pub tok_source: Vec<u8>,
    pub tok_flags: Vec<u8>,
    pub tok_hop: Vec<u8>,
    pub tok_path: Vec<u16>,
    pub tok_slot: Vec<u16>,
    pub tok_group: Vec<u16>,
    pub tok_dt: Vec<i32>,
    pub tok_lag: Vec<i32>,
    pub tok_ended: Vec<i32>,
    pub tok_month: Vec<u8>,
    pub tok_cell_start: Vec<u32>,
    pub tok_cell_count: Vec<u8>,
    pub cell_field: Vec<u16>,
    pub cell_code: Vec<u16>,
    pub cell_value: Vec<f32>,
    pub cell_raw: Vec<f32>,
    pub cell_ctype: Vec<u8>,
    pub cell_unit: Vec<u8>,
    pub cell_tok: Vec<u32>,
    pub entity: i32,
    pub as_of: Day,
    pub query: u8,
    pub horizon: u8,
    pub index_flags: u32,
    /// the sample was longer than the sequence length and lost its oldest tokens
    pub truncated: bool,
    /// corpus ids the model vocabulary does not know (mapped to unknown)
    pub unknown_ids: u64,
    pub neighbours: usize,
    pub targets: usize,
}

/// `query`: the question's position in `texts::QUERIES` (`anchors::question_query`)
#[allow(clippy::too_many_arguments)]
pub fn pack_one(
    s: &Streams,
    md: &ModelDir,
    remap: &[u32],
    opts: &Options,
    entity: i32,
    as_of: Day,
    query: u8,
    horizon: u8,
    sample: &Sample,
) -> PackedSample {
    let row = SampleRow {
        entity,
        as_of,
        split: 1,
        flags: 0,
        query,
        horizon,
        weight: 1.0,
        labels: [HorizonLabel { event: 255, censored: 0, days: -1 }; 4],
        row: 0,
    };
    let mut tokens = sample.toks.clone();
    tokens.extend_from_slice(&sample.targets);
    let m = Materialised {
        samples: vec![row],
        index: vec![IndexRow {
            tok_start: 0,
            tok_count: sample.toks.len() as u32,
            target_count: sample.targets.len() as u32,
            slot_start: 0,
            slot_count: sample.slots.len() as u32,
            extra_start: 0,
            extra_count: sample.extra.records.len() as u32,
            flags: sample.flags,
        }],
        tokens,
        extra: sample.extra.records.clone(),
        extra_cells: sample.extra.cells.clone(),
    };
    let ctx = Ctx { s, m: &m, md, remap, opts };
    let mut unknown = 0u64;
    let r = resolve(&ctx, 0, &mut unknown);
    let n = r.toks.len();
    let mut p = PackedSample {
        n_tok: n,
        entity,
        as_of,
        query,
        horizon,
        index_flags: r.index_flags,
        truncated: r.truncated,
        unknown_ids: unknown,
        neighbours: sample.neighbours,
        targets: r.targets.len(),
        ..Default::default()
    };
    // the shard writer's fill for row 0, start 0 (pack.rs write_shard)
    for (k, t) in r.toks.iter().enumerate() {
        p.tok_kind.push(t.kind);
        p.tok_source.push(t.source);
        p.tok_flags.push(t.flags);
        p.tok_hop.push(t.hop);
        p.tok_path.push(t.path);
        p.tok_slot.push(t.slot);
        p.tok_group.push(t.group);
        p.tok_dt.push(t.dt);
        p.tok_lag.push(t.lag);
        p.tok_ended.push(t.ended);
        p.tok_month.push(t.month);
        p.tok_cell_start.push(p.cell_field.len() as u32);
        p.tok_cell_count.push(t.cell_count);
        for c in &r.cells[t.cell_start as usize..t.cell_start as usize + usize::from(t.cell_count)] {
            p.cell_field.push(c.field);
            p.cell_code.push(c.code);
            p.cell_value.push(c.value);
            p.cell_raw.push(c.raw);
            p.cell_ctype.push(c.ctype);
            p.cell_unit.push(c.unit);
            p.cell_tok.push(k as u32);
        }
    }
    p
}
