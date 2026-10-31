//! the forward pass on the CPU, step for step as the network was trained: the vocabulary
//! embedding, the cells, the records and their pooling rounds, the backbone of gated delta-net and
//! attention blocks, the readout and the hazard heads. fp32 throughout, the delta rule token by
//! token, the exact erf GELU; about a second per answer on a desktop. the
//! matrix products go through `matrixmultiply`'s sgemm, split over the output's columns on rayon
//! (one company's sample is a few hundred rows, too few to split by row)

use std::collections::HashSet;
use std::path::Path;

use rayon::prelude::*;
use safetensors::SafeTensors;

use crate::artefacts::{Artefacts, ModelConfig};
use crate::batch::HostBatch;
use crate::error::{Error, Result};
use crate::outputs::Outputs;

const RMS_EPS_F32: f32 = 1.192_092_9e-7;
const LN_EPS: f32 = 1e-5;
const GDN_NORM_EPS: f32 = 1e-5;

struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    out: usize,
    inp: usize,
}

struct Nrm {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
}

struct Round {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    norm_h: Nrm,
    mlp_h: (Lin, Lin),
    back: Lin,
    norm_c: Nrm,
    mlp_c: (Lin, Lin),
}

struct Gdn {
    /// q, k, v, gate, a, b as one [4d + 2H, d] weight: one product per block
    proj: Lin,
    /// [3d, 4]
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
    o_proj: Lin,
}

enum Mixer {
    Gdn(Box<Gdn>),
    Attn { qkv: Lin, o: Lin },
}

struct Block {
    norm1: Nrm,
    norm2: Nrm,
    mixer: Mixer,
    up: Lin,
    down: Lin,
}

struct W {
    vocab_residual: Vec<f32>,
    vocab_proj: Lin,
    field_proj: Lin,
    code_proj: Lin,
    cell_num: Lin,
    ctype: Vec<f32>,
    unit: Vec<f32>,
    cell_norm: Nrm,
    cell_mlp: (Lin, Lin),
    kind_proj: Lin,
    path_proj: Lin,
    source: Vec<f32>,
    hop: Vec<f32>,
    flags: Lin,
    time_dt: Lin,
    time_lag: Lin,
    time_ended: Lin,
    month: Vec<f32>,
    ended_flag: Vec<f32>,
    rounds: Vec<Round>,
    out_norm: Nrm,
    blocks: Vec<Block>,
    final_norm: Nrm,
    state_wq: Vec<f32>,
    readout_ln: Nrm,
    readout_1: Lin,
    readout_2: Lin,
    eco_proj: Option<Lin>,
    survival: (Lin, Lin),
    early: (Lin, Lin),
}

struct Loader<'a> {
    st: SafeTensors<'a>,
    used: HashSet<String>,
}

impl Loader<'_> {
    fn t(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.st.tensor(name).map_err(|e| Error::Weights(format!("{name}: {e}")))?;
        if t.dtype() != safetensors::Dtype::F32 {
            return Err(Error::Weights(format!("{name}: dtype {:?}, fp32 expected", t.dtype())));
        }
        if t.shape() != shape {
            return Err(Error::Weights(format!("{name}: shape {:?}, expected {:?}", t.shape(), shape)));
        }
        self.used.insert(name.to_owned());
        Ok(t.data().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    fn lin(&mut self, prefix: &str, out: usize, inp: usize, bias: bool) -> Result<Lin> {
        let w = self.t(&format!("{prefix}.weight"), &[out, inp])?;
        let b = if bias { Some(self.t(&format!("{prefix}.bias"), &[out])?) } else { None };
        Ok(Lin { w, b, out, inp })
    }

    fn norm(&mut self, prefix: &str, d: usize, bias: bool) -> Result<Nrm> {
        Ok(Nrm { w: self.t(&format!("{prefix}.weight"), &[d])?, b: if bias { Some(self.t(&format!("{prefix}.bias"), &[d])?) } else { None } })
    }

    fn mlp(&mut self, prefix: &str, d: usize, mult: usize) -> Result<(Lin, Lin)> {
        Ok((self.lin(&format!("{prefix}.fc1"), d * mult, d, true)?, self.lin(&format!("{prefix}.fc2"), d, d * mult, true)?))
    }
}

/// the checkpoint by its tensors' names and shapes, into memory; a missing name or a wrong shape
/// refuses the load
fn load(path: &Path, cfg: &ModelConfig) -> Result<W> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the checkpoint is read once while mapped and copied out; nothing writes it meanwhile
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap).map_err(|e| Error::Weights(e.to_string()))?;
    let mut l = Loader { st, used: HashSet::new() };
    let (v, td, d, dc, dz, hm) = (cfg.vocab_size, cfg.text_dim, cfg.d_model, cfg.d_cell, cfg.d_z, cfg.max_horizon_months);
    let ff = cfg.fourier_freqs;
    let mut rounds = Vec::new();
    for r in 0..cfg.pool_rounds {
        rounds.push(Round {
            q: l.lin(&format!("records.q.{r}"), d, d, false)?,
            k: l.lin(&format!("records.k.{r}"), d, dc, false)?,
            v: l.lin(&format!("records.v.{r}"), d, dc, false)?,
            o: l.lin(&format!("records.o.{r}"), d, d, false)?,
            norm_h: l.norm(&format!("records.norm_h.{r}"), d, true)?,
            mlp_h: l.mlp(&format!("records.mlp_h.{r}"), d, 2)?,
            back: l.lin(&format!("records.back.{r}"), dc, d, false)?,
            norm_c: l.norm(&format!("records.norm_c.{r}"), dc, true)?,
            mlp_c: l.mlp(&format!("records.mlp_c.{r}"), dc, 2)?,
        });
    }
    let gh = cfg.gdn_heads();
    let mut blocks = Vec::new();
    for i in 0..cfg.n_blocks {
        let p = format!("backbone.blocks.{i}");
        let mixer = if cfg.is_attention_block(i) {
            Mixer::Attn { qkv: l.lin(&format!("{p}.qkv"), 3 * d, d, false)?, o: l.lin(&format!("{p}.o"), d, d, false)? }
        } else {
            let mut w = Vec::new();
            for (name, rows) in [("q_proj", d), ("k_proj", d), ("v_proj", d), ("g_proj", d), ("a_proj", gh), ("b_proj", gh)] {
                w.extend(l.t(&format!("{p}.mixer.{name}.weight"), &[rows, d])?);
            }
            let mut conv = Vec::new();
            for name in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                conv.extend(l.t(&format!("{p}.mixer.{name}.weight"), &[d, 1, 4])?);
            }
            Mixer::Gdn(Box::new(Gdn {
                proj: Lin { w, b: None, out: 4 * d + 2 * gh, inp: d },
                conv,
                a_log: l.t(&format!("{p}.mixer.A_log"), &[gh])?,
                dt_bias: l.t(&format!("{p}.mixer.dt_bias"), &[gh])?,
                o_norm: l.t(&format!("{p}.mixer.o_norm.weight"), &[cfg.gdn_head_dim])?,
                o_proj: l.lin(&format!("{p}.mixer.o_proj"), d, d, false)?,
            }))
        };
        blocks.push(Block {
            norm1: l.norm(&format!("{p}.norm1"), d, false)?,
            norm2: l.norm(&format!("{p}.norm2"), d, false)?,
            mixer,
            up: l.lin(&format!("{p}.mlp.up"), 2 * d * cfg.mlp_mult, d, false)?,
            down: l.lin(&format!("{p}.mlp.down"), d, d * cfg.mlp_mult, false)?,
        });
    }
    // the readout's state scores through (W_kᵀ q), formed once
    let wk = l.t("readout.state_k.weight", &[d, d])?;
    let sq = l.t("readout.state_q", &[d])?;
    let mut state_wq = vec![0f32; d];
    for (o, row) in wk.chunks_exact(d).enumerate() {
        for (i, w) in row.iter().enumerate() {
            state_wq[i] += w * sq[o];
        }
    }
    let _ = l.t("cells.mask_vec", &[dc])?;
    Ok(W {
        vocab_residual: l.t("vocab.residual.weight", &[v, d])?,
        vocab_proj: l.lin("vocab.proj", d, td, false)?,
        field_proj: l.lin("cells.field_proj", dc, d, false)?,
        code_proj: l.lin("cells.code_proj", dc, d, false)?,
        cell_num: l.lin("cells.num.proj", dc, 1 + 2 * ff, true)?,
        ctype: l.t("cells.ctype.weight", &[8, dc])?,
        unit: l.t("cells.unit.weight", &[16, dc])?,
        cell_norm: l.norm("cells.norm", dc, true)?,
        cell_mlp: l.mlp("cells.mlp", dc, 2)?,
        kind_proj: l.lin("records.kind_proj", d, d, false)?,
        path_proj: l.lin("records.path_proj", d, d, false)?,
        source: l.t("records.source.weight", &[16, d])?,
        hop: l.t("records.hop.weight", &[8, d])?,
        flags: l.lin("records.flags", d, 5, false)?,
        time_dt: l.lin("records.time.dt.proj", d, 17, true)?,
        time_lag: l.lin("records.time.lag.proj", d, 17, true)?,
        time_ended: l.lin("records.time.ended.proj", d, 17, true)?,
        month: l.t("records.time.month.weight", &[13, d])?,
        ended_flag: l.t("records.time.ended_flag.weight", &[2, d])?,
        rounds,
        out_norm: l.norm("records.out_norm", d, true)?,
        blocks,
        final_norm: l.norm("backbone.norm", d, false)?,
        state_wq,
        readout_ln: l.norm("readout.mlp.0", 3 * d, true)?,
        readout_1: l.lin("readout.mlp.1", dz, 3 * d, true)?,
        readout_2: l.lin("readout.mlp.3", dz, dz, true)?,
        eco_proj: if cfg.eco_dim > 0 { Some(l.lin("eco_proj", d, cfg.eco_dim, false)?) } else { None },
        survival: (l.lin("survival.net.0", dz, dz, true)?, l.lin("survival.net.2", hm * (cfg.n_causes + 1), dz, true)?),
        early: (l.lin("early.net.0", dz, dz, true)?, l.lin("early.net.2", hm * 3, dz, true)?),
    })
}

// ------------------------------------------------------------ the operations

/// a pointer the column blocks of one product share; each block writes its own columns only
struct Cols(*mut f32);
// SAFETY: the blocks of `gemm` write disjoint columns of one matrix the caller holds mutably
unsafe impl Send for Cols {}
unsafe impl Sync for Cols {}
impl Cols {
    fn at(&self, j: usize) -> *mut f32 {
        self.0.wrapping_add(j)
    }
}

/// c[m, n] = x[m, k] · w[n, k]ᵀ + beta · c, over column blocks
fn gemm(x: &[f32], w: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, beta: f32) {
    const BLOCK: usize = 64;
    if m == 0 || n == 0 {
        return;
    }
    debug_assert!(x.len() >= m * k && w.len() >= n * k && c.len() >= m * n);
    let cp = Cols(c.as_mut_ptr());
    (0..n.div_ceil(BLOCK)).into_par_iter().for_each(|blk| {
        let j0 = blk * BLOCK;
        let nn = BLOCK.min(n - j0);
        // SAFETY: A is x [m, k] row-major, B = wᵀ over rows j0.. of w (k × nn, column stride k),
        // C the columns j0..j0 + nn of c [m, n]; the bounds were checked above
        unsafe {
            matrixmultiply::sgemm(m, k, nn, 1.0, x.as_ptr(), k as isize, 1, w.as_ptr().add(j0 * k), 1, k as isize, beta, cp.at(j0), n as isize, 1);
        }
    });
}

fn add_bias(y: &mut [f32], b: &[f32]) {
    for row in y.chunks_exact_mut(b.len()) {
        for (v, bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
}

/// y = x · Wᵀ + b
fn linear(x: &[f32], rows: usize, l: &Lin) -> Vec<f32> {
    let mut y = vec![0f32; rows * l.out];
    linear_into(x, rows, l, &mut y, 0.0);
    y
}

/// y = x · Wᵀ + b + beta · y
fn linear_into(x: &[f32], rows: usize, l: &Lin, y: &mut [f32], beta: f32) {
    if rows == 0 {
        return;
    }
    gemm(x, &l.w, y, rows, l.out, l.inp, beta);
    if let Some(b) = &l.b {
        add_bias(&mut y[..rows * l.out], b);
    }
}

fn add(y: &mut [f32], x: &[f32]) {
    for (a, b) in y.iter_mut().zip(x) {
        *a += b;
    }
}

/// y[r, :] += x[r, :] · m[r]
fn add_rows_scaled(y: &mut [f32], x: &[f32], m: &[f32], d: usize) {
    for ((yr, xr), &w) in y.chunks_exact_mut(d).zip(x.chunks_exact(d)).zip(m) {
        for (a, b) in yr.iter_mut().zip(xr) {
            *a += b * w;
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// out[r, :] = table[ids[r], :] (a negative id reads row 0, the pad row)
fn gather(table: &[f32], ids: &[i32], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; ids.len() * d];
    for (o, &id) in out.chunks_exact_mut(d).zip(ids) {
        let r = id.max(0) as usize;
        o.copy_from_slice(&table[r * d..(r + 1) * d]);
    }
    out
}

/// out[r, :] += table[ids[r], :] · (m[r] or 1)
fn gather_add(out: &mut [f32], table: &[f32], ids: &[i32], m: Option<&[f32]>, d: usize) {
    for (r, (o, &id)) in out.chunks_exact_mut(d).zip(ids).enumerate() {
        let w = m.map_or(1.0, |m| m[r]);
        let row = &table[id.max(0) as usize * d..][..d];
        for (a, b) in o.iter_mut().zip(row) {
            *a += b * w;
        }
    }
}

fn layernorm(x: &[f32], d: usize, n: &Nrm) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (xr, yr) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        let mean = xr.iter().sum::<f32>() / d as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for (i, (o, v)) in yr.iter_mut().zip(xr).enumerate() {
            *o = (v - mean) * inv * n.w[i] + n.b.as_ref().map_or(0.0, |b| b[i]);
        }
    }
    y
}

fn rmsnorm(x: &[f32], d: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (xr, yr) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
        let inv = 1.0 / (xr.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for (i, (o, v)) in yr.iter_mut().zip(xr).enumerate() {
            *o = v * inv * w[i];
        }
    }
    y
}

/// fc2(gelu(fc1(x)))
fn mlp(x: &[f32], rows: usize, (fc1, fc2): &(Lin, Lin)) -> Vec<f32> {
    let mut h = linear(x, rows, fc1);
    h.iter_mut().for_each(|v| *v = gelu(*v));
    linear(&h, rows, fc2)
}

/// the Fourier number map [x, sin(x f), cos(x f)] of a scalar per row through its linear
fn fourier(x: &[f32], freqs: &[f32], l: &Lin) -> Vec<f32> {
    let width = 1 + 2 * freqs.len();
    let mut feats = vec![0f32; x.len() * width];
    for (o, &v) in feats.chunks_exact_mut(width).zip(x) {
        o[0] = v;
        for (j, f) in freqs.iter().enumerate() {
            let (s, c) = (v * f).sin_cos();
            o[1 + j] = s;
            o[1 + freqs.len() + j] = c;
        }
    }
    linear(&feats, x.len(), l)
}

fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|j| 10f64.powf(lo + (hi - lo) * j as f64 / (n.max(2) - 1) as f64)).collect()
}

pub struct CpuEngine {
    pub a: Artefacts,
    w: W,
    freqs_cell: Vec<f32>,
    freqs_time: Vec<f32>,
    inv_period: Vec<f32>,
}

impl CpuEngine {
    pub fn new(a: Artefacts, weights: &Path) -> Result<CpuEngine> {
        let w = load(weights, &a.cfg)?;
        let two_pi = std::f64::consts::PI * 2.0;
        let freqs_cell = logspace((1.0f64 / 16.0).log10(), 8f64.log10(), a.cfg.fourier_freqs).iter().map(|f| (f * two_pi) as f32).collect();
        let freqs_time = logspace((1.0f64 / 16.0).log10(), 8f64.log10(), 8).iter().map(|f| (f * two_pi) as f32).collect();
        let dh = a.cfg.d_model / a.cfg.n_heads;
        let inv_period = logspace(a.cfg.rope_min_days.log10(), a.cfg.rope_max_days.log10(), dh / 2).iter().map(|p| (two_pi / p) as f32).collect();
        tracing::info!(threads = rayon::current_num_threads(), "CPU engine ready");
        Ok(CpuEngine { a, w, freqs_cell, freqs_time, inv_period })
    }

    /// the vocabulary embedding: residual[id] + texts[id] · projᵀ
    fn embed(&self, ids: &[i32]) -> Vec<f32> {
        let mut out = gather(&self.w.vocab_residual, ids, self.a.cfg.d_model);
        let t = gather(&self.a.texts, ids, self.a.cfg.text_dim);
        linear_into(&t, ids.len(), &self.w.vocab_proj, &mut out, 1.0);
        out
    }

    /// the forward pass of a batch
    pub fn forward(&self, b: &HostBatch) -> Result<Outputs> {
        let cfg = &self.a.cfg;
        let w = &self.w;
        let (d, dc) = (cfg.d_model, cfg.d_cell);
        let (n, c, s_n) = (b.n_tok, b.cells, b.samples);

        // ---- cells [C, dc]
        let mut cells = {
            let ef = self.embed(&b.cell_field);
            let mut x = linear(&ef, c, &w.field_proj);
            gather_add(&mut x, &w.ctype, &b.cell_ctype, None, dc);
            gather_add(&mut x, &w.unit, &b.unit_c, None, dc);
            let ec = self.embed(&b.cell_code);
            let xc = linear(&ec, c, &w.code_proj);
            add_rows_scaled(&mut x, &xc, &b.has_code, dc);
            let num = fourier(&b.cell_value, &self.freqs_cell, &w.cell_num);
            add_rows_scaled(&mut x, &num, &b.is_numlike, dc);
            let mut xn = layernorm(&x, dc, &w.cell_norm);
            let m = mlp(&xn, c, &w.cell_mlp);
            add(&mut xn, &m);
            xn
        };

        // ---- record heads [N, d]
        let mut h = {
            let ek = self.embed(&b.tok_kind);
            let mut x = linear(&ek, n, &w.kind_proj);
            let ep = self.embed(&b.tok_path);
            linear_into(&ep, n, &w.path_proj, &mut x, 1.0);
            gather_add(&mut x, &w.source, &b.source_c, None, d);
            gather_add(&mut x, &w.hop, &b.hop_c, None, d);
            let bits: Vec<f32> = b.tok_flags.iter().flat_map(|&f| (0..5).map(move |i| ((f >> i) & 1) as f32)).collect();
            linear_into(&bits, n, &w.flags, &mut x, 1.0);
            let f_dt: Vec<f32> = b.tok_dt.iter().map(|&v| (v as f32 / 30.0).asinh()).collect();
            let f_lag: Vec<f32> = b.tok_lag.iter().map(|&v| (v as f32 / 30.0).asinh()).collect();
            let f_end: Vec<f32> = b.tok_ended.iter().map(|&v| (v.max(0) as f32 / 30.0).asinh()).collect();
            let is_end: Vec<f32> = b.tok_ended.iter().map(|&v| if v >= 0 { 1.0 } else { 0.0 }).collect();
            let is_end_ids: Vec<i32> = b.tok_ended.iter().map(|&v| i32::from(v >= 0)).collect();
            add(&mut x, &fourier(&f_dt, &self.freqs_time, &w.time_dt));
            add(&mut x, &fourier(&f_lag, &self.freqs_time, &w.time_lag));
            add_rows_scaled(&mut x, &fourier(&f_end, &self.freqs_time, &w.time_ended), &is_end, d);
            gather_add(&mut x, &w.month, &b.tok_month, None, d);
            gather_add(&mut x, &w.ended_flag, &is_end_ids, None, d);
            x
        };

        // ---- head ↔ cells rounds: per head, a softmax over each record's own cells
        let (ph, pdh) = (cfg.pool_heads, d / cfg.pool_heads);
        for r in &w.rounds {
            let q = linear(&h, n, &r.q);
            let k = linear(&cells, c, &r.k);
            let v = linear(&cells, c, &r.v);
            let scale = 1.0 / (pdh as f32).sqrt();
            let mut scores = vec![0f32; c * ph];
            for (ci, sc) in scores.chunks_exact_mut(ph).enumerate() {
                let t = b.cell_tok[ci] as usize;
                for (hh, s) in sc.iter_mut().enumerate() {
                    let qa = &q[(t * ph + hh) * pdh..][..pdh];
                    let ka = &k[(ci * ph + hh) * pdh..][..pdh];
                    *s = qa.iter().zip(ka).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
            }
            let mut pooled = vec![0f32; n * d];
            for (t, out) in pooled.chunks_exact_mut(d).enumerate() {
                let (c0, cnt) = (b.tok_cell_start[t] as usize, b.tok_cell_count[t] as usize);
                if cnt == 0 {
                    continue;
                }
                for hh in 0..ph {
                    let m = (0..cnt).map(|j| scores[(c0 + j) * ph + hh]).fold(-1e4f32, f32::max);
                    let mut den = 0f32;
                    let o = &mut out[hh * pdh..(hh + 1) * pdh];
                    for j in 0..cnt {
                        let e = (scores[(c0 + j) * ph + hh] - m).exp();
                        den += e;
                        for (a, vv) in o.iter_mut().zip(&v[(c0 + j) * d + hh * pdh..][..pdh]) {
                            *a += e * vv;
                        }
                    }
                    let den = den.max(1e-9);
                    o.iter_mut().for_each(|a| *a /= den);
                }
            }
            linear_into(&pooled, n, &r.o, &mut h, 1.0);
            let nh = layernorm(&h, d, &r.norm_h);
            add(&mut h, &mlp(&nh, n, &r.mlp_h));
            // the cells receive the head as normalised after its MLP update
            let nh2 = layernorm(&h, d, &r.norm_h);
            let back = linear(&nh2, n, &r.back);
            gather_add(&mut cells, &back, &b.cell_tok, None, dc);
            let nc = layernorm(&cells, dc, &r.norm_c);
            add(&mut cells, &mlp(&nc, c, &r.mlp_c));
        }
        let mut x = layernorm(&h, d, &w.out_norm);
        for (row, &v) in x.chunks_exact_mut(d).zip(&b.valid) {
            row.iter_mut().for_each(|a| *a *= v);
        }

        // ---- the economy state on the query token
        if let (Some(eco), Some(proj)) = (&self.a.economy, &w.eco_proj) {
            let rows = gather(&eco.state, &b.eco_row, cfg.eco_dim);
            let e = linear(&rows, s_n, proj);
            gather_add(&mut x, &e, &b.sample_clamped, Some(&b.is_query), d);
        }

        // ---- backbone
        let (h_all, h_slot) = cfg.head_split();
        let (hh, dh) = (cfg.n_heads, d / cfg.n_heads);
        let (gh, gd) = (cfg.gdn_heads(), cfg.gdn_head_dim);
        let l = b.seq_len;
        for blk in &w.blocks {
            let hn = rmsnorm(&x, d, &blk.norm1.w, RMS_EPS_F32);
            match &blk.mixer {
                Mixer::Gdn(m) => {
                    let ldp = m.proj.out;
                    let proj = linear(&hn, n, &m.proj);
                    // causal depthwise convolution (kernel 4) and SiLU over q, k, v, within a segment
                    let mut conv = vec![0f32; n * 3 * d];
                    for t in 0..n {
                        for ch in 0..3 * d {
                            let mut a = 0f32;
                            for j in 0..4 {
                                let tt = t as isize - 3 + j as isize;
                                if tt >= 0 && b.seg_id[tt as usize] == b.seg_id[t] {
                                    a += m.conv[ch * 4 + j] * proj[tt as usize * ldp + ch];
                                }
                            }
                            conv[t * 3 * d + ch] = silu(a);
                        }
                    }
                    // q ← q/‖q‖/√D, k ← k/‖k‖ per (token, head)
                    let qscale = 1.0 / (gd as f32).sqrt();
                    for t in 0..n {
                        for hd in 0..gh {
                            for (off, s) in [(0, qscale), (d, 1.0)] {
                                let seg = &mut conv[t * 3 * d + off + hd * gd..][..gd];
                                let inv = 1.0 / (seg.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt() * s;
                                seg.iter_mut().for_each(|v| *v *= inv);
                            }
                        }
                    }
                    // the decay g = −exp(A_log) · softplus(a + dt_bias) and the write strength β = sigmoid(b)
                    let mut g = vec![0f32; n * gh];
                    let mut beta = vec![0f32; n * gh];
                    for t in 0..n {
                        for hd in 0..gh {
                            g[t * gh + hd] = -m.a_log[hd].exp() * softplus(proj[t * ldp + 4 * d + hd] + m.dt_bias[hd]);
                            beta[t * gh + hd] = sigmoid(proj[t * ldp + 4 * d + gh + hd]);
                        }
                    }
                    // the gated delta rule per (segment, head), S[i][j] from zero:
                    //   S ← exp(g) S ; u = Sᵀk ; S += k ⊗ β(v − u) ; o = Sᵀq
                    let mut o = vec![0f32; n * d];
                    let heads: Vec<Vec<f32>> = (0..gh)
                        .into_par_iter()
                        .map(|hd| {
                            let mut out = vec![0f32; n * gd];
                            for sgi in 0..b.segments.len() - 1 {
                                let (t0, t1) = (b.segments[sgi] as usize, b.segments[sgi + 1] as usize);
                                let mut s = vec![0f32; gd * gd];
                                for t in t0..t1 {
                                    let row = &conv[t * 3 * d..];
                                    let (q, k, v) = (&row[hd * gd..][..gd], &row[d + hd * gd..][..gd], &row[2 * d + hd * gd..][..gd]);
                                    let (gt, bt) = (g[t * gh + hd].exp(), beta[t * gh + hd]);
                                    s.iter_mut().for_each(|x| *x *= gt);
                                    for j in 0..gd {
                                        let u: f32 = (0..gd).map(|i| s[i * gd + j] * k[i]).sum();
                                        let delta = bt * (v[j] - u);
                                        for i in 0..gd {
                                            s[i * gd + j] += k[i] * delta;
                                        }
                                        out[t * gd + j] = (0..gd).map(|i| s[i * gd + j] * q[i]).sum();
                                    }
                                }
                            }
                            out
                        })
                        .collect();
                    for (hd, ho) in heads.iter().enumerate() {
                        for t in 0..n {
                            o[t * d + hd * gd..][..gd].copy_from_slice(&ho[t * gd..][..gd]);
                        }
                    }
                    // per (token, head): rms norm over D, × w, × silu(gate)
                    let mut on = vec![0f32; n * d];
                    for t in 0..n {
                        for hd in 0..gh {
                            let src = &o[t * d + hd * gd..][..gd];
                            let inv = 1.0 / (src.iter().map(|v| v * v).sum::<f32>() / gd as f32 + GDN_NORM_EPS).sqrt();
                            for i in 0..gd {
                                on[t * d + hd * gd + i] = src[i] * inv * m.o_norm[i] * silu(proj[t * ldp + 3 * d + hd * gd + i]);
                            }
                        }
                    }
                    linear_into(&on, n, &m.o_proj, &mut x, 1.0);
                }
                Mixer::Attn { qkv, o } => {
                    let mut qkv_v = linear(&hn, n, qkv);
                    // rotary time on q and k by the token's day: [t, 3, H, dh], pairs (p, p + dh/2)
                    let half = dh / 2;
                    for t in 0..n {
                        for grp in 0..2 {
                            for hd in 0..hh {
                                let base = ((t * 3 + grp) * hh + hd) * dh;
                                for p in 0..half {
                                    let (s, co) = (b.days[t] * self.inv_period[p]).sin_cos();
                                    let (x1, x2) = (qkv_v[base + p], qkv_v[base + p + half]);
                                    qkv_v[base + p] = x1 * co - x2 * s;
                                    qkv_v[base + p + half] = x1 * s + x2 * co;
                                }
                            }
                        }
                    }
                    let scale = 1.0 / (dh as f32).sqrt();
                    let mut merged = vec![0f32; n * d];
                    for r in 0..b.rows {
                        let rows: Vec<Vec<f32>> = (0..hh)
                            .into_par_iter()
                            .map(|hd| {
                                let at = |t: usize, grp: usize| &qkv_v[((r * l + t) * 3 + grp) * hh * dh + hd * dh..][..dh];
                                let mut out = vec![0f32; l * dh];
                                let mut sc = vec![0f32; l];
                                for qi in 0..l {
                                    let tq = r * l + qi;
                                    let (sq, slq, gq) = (b.tok_sample[tq], b.tok_slot[tq], b.tok_group[tq]);
                                    let qv = at(qi, 0);
                                    let mut m = f32::NEG_INFINITY;
                                    for (ki, s) in sc.iter_mut().enumerate() {
                                        let tk = r * l + ki;
                                        // the mask: the same sample, and per head group the whole
                                        // sample, the same slot, or the focal slot and the same case
                                        let mut ok = ki == qi;
                                        if !ok && sq >= 0 && b.tok_sample[tk] == sq {
                                            ok = if hd < h_all {
                                                true
                                            } else if hd < h_all + h_slot {
                                                b.tok_slot[tk] == slq
                                            } else {
                                                b.tok_slot[tk] == 0 || (gq > 0 && b.tok_group[tk] == gq) || b.tok_slot[tk] == slq
                                            };
                                        }
                                        *s = if ok { qv.iter().zip(at(ki, 1)).map(|(a, b)| a * b).sum::<f32>() * scale } else { f32::NEG_INFINITY };
                                        m = m.max(*s);
                                    }
                                    let mut sum = 0f32;
                                    for s in sc.iter_mut() {
                                        *s = if *s == f32::NEG_INFINITY { 0.0 } else { (*s - m).exp() };
                                        sum += *s;
                                    }
                                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                                    let o = &mut out[qi * dh..][..dh];
                                    for (ki, &p) in sc.iter().enumerate() {
                                        if p != 0.0 {
                                            for (a, vv) in o.iter_mut().zip(at(ki, 2)) {
                                                *a += p * inv * vv;
                                            }
                                        }
                                    }
                                }
                                out
                            })
                            .collect();
                        for (hd, ho) in rows.iter().enumerate() {
                            for qi in 0..l {
                                merged[(r * l + qi) * d + hd * dh..][..dh].copy_from_slice(&ho[qi * dh..][..dh]);
                            }
                        }
                    }
                    linear_into(&merged, n, o, &mut x, 1.0);
                }
            }
            let hn2 = rmsnorm(&x, d, &blk.norm2.w, RMS_EPS_F32);
            let up = linear(&hn2, n, &blk.up);
            let hid = blk.up.out / 2;
            // SwiGLU: up[:, c] · silu(up[:, hid + c])
            let mut sw = vec![0f32; n * hid];
            for (o, u) in sw.chunks_exact_mut(hid).zip(up.chunks_exact(2 * hid)) {
                for cc in 0..hid {
                    o[cc] = u[cc] * silu(u[hid + cc]);
                }
            }
            linear_into(&sw, n, &blk.down, &mut x, 1.0);
        }
        let x = rmsnorm(&x, d, &w.final_norm.w, RMS_EPS_F32);

        // ---- readout: the query token, attention over the state tokens, the mean of the focal tokens
        let sd = (d as f32).sqrt();
        let scores: Vec<f32> = x.chunks_exact(d).map(|r| r.iter().zip(&w.state_wq).map(|(a, b)| a * b).sum::<f32>() / sd).collect();
        let mut pooled = vec![0f32; s_n * 3 * d];
        for (s, out) in pooled.chunks_exact_mut(3 * d).enumerate() {
            let (a, e) = (b.smp_tok_begin[s] as usize, b.smp_tok_end[s] as usize);
            let toks = a..e.max(a);
            let m = toks.clone().filter(|&t| b.is_state[t] > 0.5).map(|t| scores[t]).fold(-1e4f32, f32::max);
            let cq = toks.clone().map(|t| b.is_query[t]).sum::<f32>().max(1.0);
            let cf = toks.clone().map(|t| b.is_focal[t]).sum::<f32>().max(1.0);
            let ws: Vec<f32> = toks.clone().map(|t| if b.is_state[t] > 0.5 { (scores[t] - m).exp() } else { 0.0 }).collect();
            let inv = 1.0 / ws.iter().sum::<f32>().max(1e-9);
            for (i, t) in toks.enumerate() {
                let (wq, wf, wsv) = (b.is_query[t] / cq, b.is_focal[t] / cf, ws[i]);
                for (cc, v) in x[t * d..(t + 1) * d].iter().enumerate() {
                    out[cc] += v * wq;
                    out[d + cc] += v * wsv;
                    out[2 * d + cc] += v * wf;
                }
            }
            out[d..2 * d].iter_mut().for_each(|v| *v *= inv);
        }
        let ln = layernorm(&pooled, 3 * d, &w.readout_ln);
        let mut z1 = linear(&ln, s_n, &w.readout_1);
        z1.iter_mut().for_each(|v| *v = gelu(*v));
        let z = linear(&z1, s_n, &w.readout_2);

        // ---- heads
        let mut hs = linear(&z, s_n, &w.survival.0);
        hs.iter_mut().for_each(|v| *v = gelu(*v));
        let surv = linear(&hs, s_n, &w.survival.1);
        let mut he = linear(&z, s_n, &w.early.0);
        he.iter_mut().for_each(|v| *v = gelu(*v));
        let early = linear(&he, s_n, &w.early.1);
        Ok(Outputs { samples: s_n, months: cfg.max_horizon_months, causes: cfg.n_causes, surv, early, z, peak_bytes: 0 })
    }
}
