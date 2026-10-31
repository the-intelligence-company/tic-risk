//! one answer: a company at an as-of day for a question, from its view (`view.rs`) through the
//! sampler, the engine and the release's calibration and bands

use std::fmt::Write as _;

use chrono::NaiveDate;
use risk_eval::question::Question;
use risk_model::live::PackedSample;
use risk_tokens::vocab::Vocab;
use serde::Serialize;

use crate::calibration::{self, Bands, Calibrator};
use crate::error::Result;
use crate::live::{Live, Sampled};
use crate::forward::Forward;
use crate::outputs::{early_cause, query_causes};
use crate::view::Fetched;

/// the release's questions
pub const QUESTIONS: [&str; 5] = ["bankruptcy_12m", "bankruptcy_24m", "enforcement_debt_12m", "fskatt_revoked_12m", "report_overdue_12m"];

/// a question with its release files
pub struct Asked {
    pub q: &'static Question,
    pub cal: Option<Calibrator>,
    pub bands: Option<Bands>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    pub company: i32,
    pub as_of: String,
    pub question: &'static str,
    pub horizon_months: u32,
    /// the model's probability before calibration
    pub p_raw: f64,
    /// calibrated: the point-in-time probability
    pub p_pit: Option<f64>,
    pub band90: Option<[f64; 2]>,
    /// 1–5 and 0–100: the 12-month bankruptcy question's scales, so that question's only
    pub risk_class: Option<u8>,
    pub score: Option<u8>,
    /// the question's raw cumulative probability at 3, 6, 12 and 24 months
    pub term_3_6_12_24_raw: Vec<f64>,
    pub tokens: usize,
    pub cells: usize,
    pub neighbours: usize,
    /// the sample was longer than the model's context and lost its oldest tokens
    pub truncated: bool,
    /// codes the model's vocabulary does not know (read as unknown)
    pub unknown_ids: u64,
}

/// `graph`: let an engine that can replay a prepared plan for the sample's shape do so
pub fn answer(engine: &dyn Forward, live: &Live, f: &Fetched, day: NaiveDate, a: &Asked, graph: bool) -> Result<(Answer, Sampled)> {
    let q = a.q;
    let art = engine.artefacts();
    let sampled = live.sample(f, day, q, &art.vocab, art.economy.as_ref())?;
    let out = engine.run(&sampled.batch, graph)?;
    let key = art.vocab.query_keys.get(sampled.packed.query as usize).map(String::as_str).unwrap_or("any_default");
    let p = f64::from(out.query_probability(0, key, q.horizon_months as usize));
    let pit = a.cal.as_ref().map(|c| c.pit(p));
    // around the calibrated probability: the bands were fitted on calibrated probabilities
    let band = a.bands.as_ref().map(|b| b.apply(pit.unwrap_or(p)));
    let headline = pit.filter(|_| q.name == "bankruptcy_12m");
    // the question's own head: an early warning's hazards, else the default causes' incidence
    let horizons = [3usize, 6, 12, 24].into_iter().filter(|&h| h <= out.months);
    let term = match early_cause(key) {
        Some(c) => {
            let early = out.early_p(0);
            horizons.map(|h| f64::from(early[(h - 1) * 3 + c])).collect()
        }
        None => {
            let cif = out.cif(0);
            horizons.map(|h| query_causes(key).iter().map(|&c| f64::from(cif[(h - 1) * out.causes + (c - 1)])).sum()).collect()
        }
    };
    let ans = Answer {
        company: f.focal,
        as_of: day.to_string(),
        question: q.name,
        horizon_months: q.horizon_months,
        p_raw: p,
        p_pit: pit,
        band90: band.map(|(lo, hi)| [lo, hi]),
        risk_class: headline.map(calibration::risk_class),
        score: headline.map(calibration::score),
        term_3_6_12_24_raw: term,
        tokens: sampled.packed.n_tok,
        cells: sampled.batch.cells,
        neighbours: sampled.packed.neighbours,
        truncated: sampled.packed.truncated,
        unknown_ids: sampled.packed.unknown_ids,
    };
    Ok((ans, sampled))
}

/// a packed sample as text, one token a line (`risk-tokens show`'s layout over the model's ids):
/// slot, kind, days before the as-of day, days from happening to known, and the cells (a code's
/// key, else the value as stored)
pub fn show(p: &PackedSample, v: &Vocab, question: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "company {} · {} · {} tokens · {} neighbours", p.entity, question, p.n_tok, p.neighbours);
    for t in 0..p.n_tok {
        let _ = write!(out, "  [{:2}] {:<36} dt {:>5} lag {:>4}", p.tok_slot[t], v.key(u32::from(p.tok_kind[t])), p.tok_dt[t], p.tok_lag[t]);
        let start = p.tok_cell_start[t] as usize;
        for c in start..start + p.tok_cell_count[t] as usize {
            let field = v.key(u32::from(p.cell_field[c]));
            let field = field.rsplit_once('.').map_or(field, |(_, f)| f);
            if p.cell_code[c] > 0 {
                let code = v.key(u32::from(p.cell_code[c]));
                let _ = write!(out, " {field}={}", code.rsplit_once('=').map_or(code, |(_, c)| c));
            } else {
                let _ = write!(out, " {field}={}", p.cell_raw[c]);
            }
        }
        out.push('\n');
    }
    out
}
