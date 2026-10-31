//! the tokenizer of the credit-risk model: the vocabulary, the per-entity record streams (records
//! with typed cells) built from the input files, the state and context records derived at an
//! as-of day, the anchor sets with their labels, and the temporal neighbour sampler. the sampler
//! is the only code that applies `known <= T`.
//!
//! every date is a day number (days since 1970-01-01), shared with `risk-eval`.

pub mod anchors;
pub mod build;
pub mod context;
pub mod fields;
pub mod record;
pub mod sampler;
pub mod state;
pub mod streams;
pub mod texts;
pub mod vocab;

pub use risk_eval::dates;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("eval: {0}")]
    Eval(#[from] risk_eval::Error),
    #[error("bad record at line {line}: {msg}")]
    Record { line: u64, msg: String },
    #[error("bad cache file {0}: {1}")]
    Cache(String, String),
    #[error("{0}")]
    Other(String),
}
