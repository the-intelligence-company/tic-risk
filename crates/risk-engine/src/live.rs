//! one sample: a company's view (`view.rs`) at an as-of day for a question goes through the
//! training sampler, is packed in memory against the model's vocabulary and normalisation, and
//! becomes the batch the engine reads. this holds what the model learned (the release's
//! vocabulary and normalisation) and the sampler's budget; it holds no data.

use std::path::Path;
use std::time::Instant;

use chrono::NaiveDate;
use risk_eval::question::{self, Question};
use risk_model::live::{PackedSample, pack_one};
use risk_model::pack::Options;
use risk_model::vocab::ModelDir;
use risk_tokens::anchors::question_query;
use risk_tokens::dates::{DAY_MAX, from_naive};
use risk_tokens::sampler::{Budget, Query, build_sample};

use crate::artefacts::{Economy, Vocab};
use crate::batch::HostBatch;
use crate::error::{Error, Result};
use crate::view::Fetched;

pub struct Live {
    pub md: ModelDir,
    pub opts: Options,
    pub budget: Budget,
}

pub struct Sampled {
    pub batch: HostBatch,
    pub packed: PackedSample,
    pub question: &'static Question,
    pub sample_ms: f64,
}

impl Live {
    /// `model_dir`: vocab.json + stats.json of the release (its hash must be the checkpoint's)
    pub fn open(model_dir: &Path, vocab_hash: &str) -> Result<Live> {
        let md = ModelDir::load(model_dir).map_err(|e| Error::Artefact(model_dir.display().to_string(), e.to_string()))?;
        if md.hash != vocab_hash {
            return Err(Error::Artefact(model_dir.display().to_string(), format!("vocabulary {} is not the checkpoint's {vocab_hash}", md.hash)));
        }
        Ok(Live { md, opts: Options { corpus: "sweden".into(), ..Options::default() }, budget: Budget::default() })
    }

    /// the release's vocabulary and normalisation without a checkpoint to match (to pack samples
    /// without running the model)
    pub fn load(model_dir: &Path) -> Result<Live> {
        let md = ModelDir::load(model_dir).map_err(|e| Error::Artefact(model_dir.display().to_string(), e.to_string()))?;
        Ok(Live { md, opts: Options { corpus: "sweden".into(), ..Options::default() }, budget: Budget::default() })
    }

    pub fn question(name: &str) -> Result<&'static Question> {
        question::find(name).ok_or_else(|| Error::Other(format!("unknown prediction {name}, expected one of {}", question::QUESTIONS.iter().map(|q| q.name).collect::<Vec<_>>().join(", "))))
    }

    /// the sample of the fetched company at an as-of day for a question, as a one-row batch
    pub fn sample(&self, f: &Fetched, as_of: NaiveDate, q: &'static Question, vocab: &Vocab, economy: Option<&Economy>) -> Result<Sampled> {
        let t0 = Instant::now();
        let packed = self.pack(f, as_of, q)?;
        let batch = HostBatch::from_packed(&packed, vocab, economy);
        Ok(Sampled { batch, packed, question: q, sample_ms: t0.elapsed().as_secs_f64() * 1000.0 })
    }

    /// the sampler's sample of the fetched company, packed against the release's vocabulary
    pub fn pack(&self, f: &Fetched, as_of: NaiveDate, q: &'static Question) -> Result<PackedSample> {
        let entity = f.focal;
        let Some(row) = f.streams.entity(entity) else {
            return Err(Error::Other(format!("company {entity} has no events to score")));
        };
        let t = from_naive(as_of);
        if row.registered != risk_tokens::dates::DAY_MIN && row.registered > t {
            return Err(Error::Other(format!("company {entity} was registered after {as_of}")));
        }
        let eco = f.ctx.view(t);
        let query = Query { target: question_query(q), horizon_months: q.horizon_months };
        let sample = build_sample(&f.streams, &f.ctx, &eco, entity, t, Some(query), &self.budget, DAY_MAX);
        if sample.toks.is_empty() {
            return Err(Error::Other(format!("company {entity} has no events known on {as_of}")));
        }
        Ok(pack_one(&f.streams, &self.md, &f.remap, &self.opts, entity, t, query.target, q.horizon_months as u8, &sample))
    }
}
