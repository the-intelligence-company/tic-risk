//! one company's view as the model reads it: its and its related companies' streams, the
//! context, the lines they were built from. built from lines whoever supplies them (an input from
//! files is one, `input.rs`): one build, so the same lines always give the same answer

use risk_eval::store::{Economy, Sector};
use risk_model::vocab::ModelDir;
use risk_store::population::CompanyLife;
use risk_tokens::build::{BuildStats, MemBuild};
use risk_tokens::context::{Context, constant_coverage};
use risk_tokens::streams::Streams;
use risk_tokens::vocab::Vocab;
use serde::Serialize;

use crate::error::{Error, Result};

/// one company's view: its streams and its related companies', the context, what it was built from
pub struct Fetched {
    pub focal: i32,
    pub streams: Streams,
    pub ctx: Context,
    /// the build's vocabulary ids → the model's (a code the release never saw → `unknown`)
    pub remap: Vec<u32>,
    pub stats: FetchStats,
    /// what the streams were built from, as lines (the event and relation lines, `company,
    /// registered,ceased` per life), each list in the build's order
    pub lines: InputLines,
    pub lives: Vec<CompanyLife>,
}

/// the streams, context and vocabulary remap built from a company's lines, in the order given
pub struct View {
    pub streams: Streams,
    pub ctx: Context,
    pub remap: Vec<u32>,
    pub bstats: BuildStats,
    pub new_codes: usize,
    pub unknown_keys: usize,
}

pub fn build_view(corpus: &Vocab, md: &ModelDir, events: &[String], relations: &[String], lives: &[CompanyLife], economy: Economy, sector: Sector) -> Result<View> {
    let mut build = MemBuild::new(corpus.clone());
    build.events(events.iter().map(String::as_str)).map_err(|e| Error::Other(format!("events: {e}")))?;
    build.relations(relations.iter().map(String::as_str)).map_err(|e| Error::Other(format!("relations: {e}")))?;
    let (streams, bstats): (Streams, BuildStats) = build.finish(lives).map_err(|e| Error::Other(format!("build: {e}")))?;
    let new_codes = streams.vocab.len() - corpus.len();
    let (remap, unknown_keys) = md.remap(&streams.vocab);
    let ctx = Context::from_parts(economy, sector, &streams.vocab, constant_coverage()).map_err(|e| Error::Other(format!("context: {e}")))?;
    Ok(View { streams, ctx, remap, bstats, new_codes, unknown_keys })
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputLines {
    pub events: Vec<String>,
    pub relations: Vec<String>,
    pub lives: Vec<String>,
}

/// a life as a line of `InputLines`
pub fn life_line(l: &CompanyLife) -> String {
    format!("{},{},{}", l.company_id, l.registered.map(|d| d.to_string()).unwrap_or_default(), l.ceased.map(|d| d.to_string()).unwrap_or_default())
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FetchStats {
    /// the time the data is from (RFC 3339)
    pub read_at: String,
    pub neighbours: usize,
    pub companies: usize,
    pub events: u64,
    pub relations: u64,
    pub records: u64,
    pub cells: u64,
    /// economy series the model was not trained with (left out)
    pub unknown_series: usize,
    /// codes the training corpus never saw (they reach the model as `unknown`)
    pub new_codes: usize,
    /// vocabulary keys of the build the model has no id for
    pub unknown_keys: usize,
    /// companies whose industry a classifier inferred while the lines were read (zero from files)
    pub sector_companies: usize,
    pub ms_sector: f64,
    /// the timings of whoever read the lines, in milliseconds (zero from files)
    pub ms_first_trip: f64,
    pub ms_neighbours: f64,
    pub ms_second_trip: f64,
    pub ms_build: f64,
    /// the whole read and build
    pub ms_total: f64,
}
