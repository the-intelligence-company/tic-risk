#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("artefact {0}: {1}")]
    Artefact(String, String),
    #[error("weights: {0}")]
    Weights(String),
    #[error("shard: {0}")]
    Shard(String),
    #[error("{0}")]
    Other(String),
}

impl From<risk_model::Error> for Error {
    fn from(e: risk_model::Error) -> Self {
        Error::Shard(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
