//! the files of a release: the checkpoint's manifest
//! and vocabulary, the frozen text table, the economy state table. nothing is edited or remapped
//! here; a vocabulary that does not match is refused.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::npy;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub text_dim: usize,
    pub d_model: usize,
    pub d_cell: usize,
    pub n_heads: usize,
    pub n_blocks: usize,
    pub pattern: String,
    pub d_z: usize,
    #[serde(default = "four")]
    pub mlp_mult: usize,
    #[serde(default = "two")]
    pub pool_rounds: usize,
    #[serde(default = "four")]
    pub pool_heads: usize,
    #[serde(default = "sixteen")]
    pub fourier_freqs: usize,
    pub rope_min_days: f64,
    pub rope_max_days: f64,
    #[serde(default = "gdn")]
    pub linear_layer: String,
    #[serde(default = "sixtyfour")]
    pub gdn_head_dim: usize,
    #[serde(default = "one_f")]
    pub gdn_expand_v: f64,
    #[serde(default = "default_groups")]
    pub head_groups: [f64; 3],
    #[serde(default = "twentyfour")]
    pub max_horizon_months: usize,
    #[serde(default = "four")]
    pub n_causes: usize,
    #[serde(default)]
    pub eco_dim: usize,
    #[serde(default)]
    pub figures: Vec<String>,
    #[serde(default)]
    pub link_families: Vec<String>,
}
fn two() -> usize { 2 }
fn four() -> usize { 4 }
fn sixteen() -> usize { 16 }
fn sixtyfour() -> usize { 64 }
fn twentyfour() -> usize { 24 }
fn one_f() -> f64 { 1.0 }
fn gdn() -> String { "gdn".into() }
fn default_groups() -> [f64; 3] { [0.5, 0.25, 0.25] }

impl ModelConfig {
    /// the attention head counts of the three mask rules: all, same slot, focal-or-case
    pub fn head_split(&self) -> (usize, usize) {
        let h = self.n_heads as f64;
        ((h * self.head_groups[0]).round() as usize, (h * self.head_groups[1]).round() as usize)
    }
    pub fn gdn_heads(&self) -> usize {
        self.d_model / self.gdn_head_dim
    }
    pub fn is_attention_block(&self, i: usize) -> bool {
        let p = self.pattern.as_bytes();
        p[i % p.len()] != b'G' || self.linear_layer == "none"
    }
}

#[derive(Debug, Deserialize)]
struct ManifestConfig {
    model: ModelConfig,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub step: u64,
    pub vocab_hash: String,
    config: ManifestConfig,
    #[serde(default)]
    pub economy: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VocabEntry {
    pub id: usize,
    pub ns: String,
    pub key: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Deserialize)]
struct VocabFile {
    entries: Vec<VocabEntry>,
}

/// the vocabulary facts the forward pass needs
pub struct Vocab {
    pub size: usize,
    pub entries: Vec<VocabEntry>,
    pub query_kind: i32,
    /// per id: a state or context kind
    pub is_state_kind: Vec<u8>,
    /// query target keys by position (the rank of their code id), what `smp_query` indexes
    pub query_keys: Vec<String>,
    pub keys: HashMap<String, i32>,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Vocab> {
        let f: VocabFile = serde_json::from_slice(&std::fs::read(path)?)?;
        let bad = |m: String| Error::Artefact(path.display().to_string(), m);
        let size = f.entries.len();
        for (i, e) in f.entries.iter().enumerate() {
            if e.id != i {
                return Err(bad(format!("entry {i} has id {}", e.id)));
            }
        }
        let keys: HashMap<String, i32> = f.entries.iter().filter(|e| e.ns != "code").map(|e| (e.key.clone(), e.id as i32)).collect();
        let query_kind = *keys.get("query").ok_or_else(|| bad("no `query` kind".into()))?;
        let mut is_state_kind = vec![0u8; size];
        for e in f.entries.iter().filter(|e| e.ns == "kind") {
            if e.key.starts_with("state.") || e.key.starts_with("context.") {
                is_state_kind[e.id] = 1;
            }
        }
        let mut q: Vec<(usize, String)> = f
            .entries
            .iter()
            .filter(|e| e.ns == "code" && e.key.starts_with("query.target="))
            .map(|e| (e.id, e.key["query.target=".len()..].to_owned()))
            .collect();
        q.sort();
        Ok(Vocab { size, query_kind, is_state_kind, query_keys: q.into_iter().map(|(_, k)| k).collect(), keys, entries: f.entries })
    }
}

/// the economy state per month, and the lookup from an as-of day to its row
pub struct Economy {
    pub dim: usize,
    pub months: usize,
    /// [months + 1, dim]: the last row is zeros, read by a month outside the table
    pub state: Vec<f32>,
    /// month number since 1970-01 → row
    index: HashMap<i64, usize>,
}

impl Economy {
    pub fn load(dir: &Path) -> Result<Economy> {
        let st = npy::read_f32(&dir.join("economy_state.npy"))?;
        let months: Vec<String> = serde_json::from_slice(&std::fs::read(dir.join("economy_months.json"))?)?;
        if st.shape.len() != 2 || st.shape[0] != months.len() {
            return Err(Error::Artefact(dir.display().to_string(), format!("economy_state {:?} vs {} months", st.shape, months.len())));
        }
        let dim = st.shape[1];
        let mut index = HashMap::new();
        for (i, m) in months.iter().enumerate() {
            let (y, mo) = m.split_once('-').ok_or_else(|| Error::Artefact(dir.display().to_string(), format!("month {m}")))?;
            let y: i64 = y.parse().map_err(|_| Error::Artefact(dir.display().to_string(), format!("month {m}")))?;
            let mo: i64 = mo[..2].parse().map_err(|_| Error::Artefact(dir.display().to_string(), format!("month {m}")))?;
            index.insert((y - 1970) * 12 + (mo - 1), i);
        }
        let mut state = st.data;
        state.extend(std::iter::repeat_n(0.0f32, dim));
        Ok(Economy { dim, months: months.len(), state, index })
    }

    /// the table row of an as-of day (days since 1970-01-01); the zero row when outside
    pub fn row_of_day(&self, day: i32) -> usize {
        let (y, m, _) = civil_from_days(day as i64);
        let mnum = (y - 1970) * 12 + (m as i64 - 1);
        self.index.get(&mnum).copied().unwrap_or(self.months)
    }
}

/// days since 1970-01-01 → (year, month, day), proleptic Gregorian (Howard Hinnant's algorithm)
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub struct Artefacts {
    pub checkpoint: PathBuf,
    pub manifest: Manifest,
    pub cfg: ModelConfig,
    pub vocab: Vocab,
    /// [V, text_dim]
    pub texts: Vec<f32>,
    pub economy: Option<Economy>,
}

impl Artefacts {
    /// `checkpoint`: the step directory (model.safetensors, manifest.json, vocab.json);
    /// `model_dir`: where texts.npy and economy_se/ live (the checkpoint directory itself when
    /// a release bundles them)
    pub fn load(checkpoint: &Path, model_dir: &Path) -> Result<Artefacts> {
        // the trainer writes NaN for an objective that had no term in a batch; JSON has no NaN
        let text = std::fs::read_to_string(checkpoint.join("manifest.json"))?
            .replace(" NaN", " null")
            .replace(" -Infinity", " null")
            .replace(" Infinity", " null");
        let manifest: Manifest = serde_json::from_str(&text)?;
        let cfg = manifest.config.model.clone();
        let vocab = Vocab::load(&checkpoint.join("vocab.json"))?;
        if vocab.size != cfg.vocab_size {
            return Err(Error::Artefact(checkpoint.display().to_string(), format!("vocab.json has {} entries, the config says {}", vocab.size, cfg.vocab_size)));
        }
        let t = npy::read_f32(&model_dir.join("texts.npy"))?;
        if t.shape != [vocab.size, cfg.text_dim] {
            return Err(Error::Artefact(model_dir.display().to_string(), format!("texts.npy {:?} vs [{}, {}]", t.shape, vocab.size, cfg.text_dim)));
        }
        let economy = if cfg.eco_dim > 0 {
            let dir = model_dir.join("economy_se");
            let e = Economy::load(&dir)?;
            if e.dim != cfg.eco_dim {
                return Err(Error::Artefact(dir.display().to_string(), format!("economy dim {} vs config {}", e.dim, cfg.eco_dim)));
            }
            Some(e)
        } else {
            None
        };
        Ok(Artefacts { checkpoint: checkpoint.to_path_buf(), manifest, cfg, vocab, texts: t.data, economy })
    }
}

#[cfg(test)]
mod tests {
    use super::civil_from_days;

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_997), (2024, 10, 1));
        assert_eq!(civil_from_days(20_362), (2025, 10, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
