//! the model's data contract: one vocabulary over every corpus, and materialised samples packed
//! into fixed-shape shards that training and scoring read identically. no tokenisation,
//! normalisation or time arithmetic happens outside this crate.

pub mod live;
pub mod pack;
pub mod shard;
pub mod vocab;

pub use risk_eval::dates;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokens: {0}")]
    Tokens(#[from] risk_tokens::Error),
    #[error("bad shard {0}: {1}")]
    Shard(String, String),
    #[error("{0}")]
    Other(String),
}
