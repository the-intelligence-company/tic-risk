//! the model vocabulary: the closed tables in table order, then the union of every corpus's open
//! codes sorted by key. a corpus id maps to a model id by key; a key the model never saw maps to
//! `unknown`. `stats.json` of the first corpus is the normalisation every corpus is packed with.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Error;
use risk_tokens::fields::Unit;
use risk_tokens::vocab::{Ns, UNKNOWN, Vocab};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldNorm {
    pub field: String,
    pub unit: String,
    pub scale: f32,
    pub asinh_mean: f64,
    pub asinh_std: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct StatsFile {
    cut: String,
    fields: Vec<FieldNorm>,
}

/// (text, english text, unit) of an open entry
type OpenEntry = (String, Option<String>, Option<String>);

/// the model directory: vocabulary, normalisation, texts
pub struct ModelDir {
    pub vocab: Vocab,
    pub cut: String,
    /// per model field id: (scale, mean, std)
    pub norm: Vec<(f32, f32, f32)>,
    pub hash: String,
}

/// FNV-1a over the keys in id order: the identity of a vocabulary
pub fn vocab_hash(v: &Vocab) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for e in &v.entries {
        for b in format!("{:?}={}\n", e.ns, e.key).bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

fn unit_scale(unit: &str) -> f32 {
    match unit {
        "sek" => Unit::Sek.scale(),
        "days" => Unit::Days.scale(),
        _ => 1.0,
    }
}

impl ModelDir {
    pub fn load(dir: &Path) -> Result<ModelDir, Error> {
        let vocab = Vocab::load(&dir.join("vocab.json"))?;
        let stats: StatsFile =
            serde_json::from_str(&std::fs::read_to_string(dir.join("stats.json"))?)?;
        let by_name: HashMap<&str, &FieldNorm> =
            stats.fields.iter().map(|f| (f.field.as_str(), f)).collect();
        let mut norm = vec![(1.0f32, 0.0f32, 1.0f32); vocab.len()];
        for e in vocab.entries.iter().filter(|e| e.ns == Ns::Field) {
            norm[e.id as usize] = match by_name.get(e.key.as_str()) {
                Some(f) => (
                    f.scale,
                    f.asinh_mean as f32,
                    if f.asinh_std.is_finite() && f.asinh_std > 1e-6 {
                        f.asinh_std as f32
                    } else {
                        1.0
                    },
                ),
                None => (e.unit.as_deref().map_or(1.0, unit_scale), 0.0, 1.0),
            };
        }
        let hash = vocab_hash(&vocab);
        Ok(ModelDir {
            vocab,
            cut: stats.cut,
            norm,
            hash,
        })
    }

    /// corpus id → model id by (namespace, key); unknown keys → `UNKNOWN`
    pub fn remap(&self, corpus: &Vocab) -> (Vec<u32>, usize) {
        let mut unknown = 0;
        let map: Vec<u32> = corpus
            .entries
            .iter()
            .map(|e| match self.vocab.id(e.ns, &e.key) {
                Some(id) => id,
                None => {
                    unknown += 1;
                    UNKNOWN
                }
            })
            .collect();
        (map, unknown)
    }
}

#[derive(Debug, Serialize)]
pub struct VocabReport {
    pub corpora: Vec<String>,
    pub entries: usize,
    pub open_codes: usize,
    pub per_corpus_unknown: Vec<usize>,
    pub hash: String,
}

/// builds the model directory from one or more token caches; the first one's `stats.json` is the
/// normalisation
pub fn build(tokens_dirs: &[std::path::PathBuf], out: &Path) -> Result<VocabReport, Error> {
    let first = tokens_dirs
        .first()
        .ok_or_else(|| Error::Other("no tokens dir".into()))?;
    let mut model = Vocab::closed();
    let mut open: BTreeMap<(Ns, String), OpenEntry> = BTreeMap::new();
    let mut corpora = Vec::new();
    let mut vocabs = Vec::new();
    for dir in tokens_dirs {
        let v = Vocab::load(&dir.join("vocab.json"))?;
        for e in &v.entries {
            if model.id(e.ns, &e.key).is_none() {
                open.entry((e.ns, e.key.clone()))
                    .or_insert_with(|| (e.text.clone(), e.text_en.clone(), e.unit.clone()));
            }
        }
        corpora.push(dir.display().to_string());
        vocabs.push(v);
    }
    // fields first (series), then codes, each sorted by key: stable ids across rebuilds with the
    // same inputs
    let open_codes = open.len();
    for ns in [Ns::Field, Ns::Code] {
        for ((n, key), (text, text_en, unit)) in &open {
            if *n == ns {
                model.add(ns, key, text, text_en.as_deref(), unit.as_deref());
            }
        }
    }
    std::fs::create_dir_all(out)?;
    model.save(&out.join("vocab.json"))?;
    std::fs::copy(first.join("stats.json"), out.join("stats.json"))?;
    let mut texts = std::io::BufWriter::new(std::fs::File::create(out.join("texts.jsonl"))?);
    for e in &model.entries {
        serde_json::to_writer(
            &mut texts,
            &serde_json::json!({
                "id": e.id,
                "ns": e.ns,
                "key": e.key,
                "text": e.text,
                "text_en": e.text_en,
            }),
        )?;
        texts.write_all(b"\n")?;
    }
    texts.flush()?;
    let md = ModelDir::load(out)?;
    let per_corpus_unknown = vocabs.iter().map(|v| md.remap(v).1).collect();
    let report = VocabReport {
        corpora,
        entries: md.vocab.len(),
        open_codes,
        per_corpus_unknown,
        hash: md.hash.clone(),
    };
    std::fs::write(
        out.join("vocab_report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_changes_with_keys() {
        let a = Vocab::closed();
        let mut b = Vocab::closed();
        b.code("debt.debt_type", "20");
        assert_ne!(vocab_hash(&a), vocab_hash(&b));
        assert_eq!(vocab_hash(&a), vocab_hash(&Vocab::closed()));
    }
}
