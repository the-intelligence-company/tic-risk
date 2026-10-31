//! the questions the credit-risk model answers and the data they read. `question`: each question's
//! target, horizon and as-of rule, and the one function that gives a (company, as-of) row its
//! outcome; `points`: the monthly as-of points, the train/test splits and the negative sampling;
//! `dates`: day numbers; `cache`: the event and relation record files as fixed-size binary records
//! sorted by company; `store`: those records in memory with the company lives, the label set and
//! the economy and sector series.
//!
//! every date is a day number (days since 1970-01-01): the cache is fixed-size records and the
//! as-of logic is integer comparisons.

pub mod cache;
pub mod dates;
pub mod points;
pub mod question;
pub mod store;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bad record at line {line}: {msg}")]
    Record { line: u64, msg: String },
    #[error("bad cache file {0}: {1}")]
    Cache(String, String),
    #[error("{0}")]
    Other(String),
}
