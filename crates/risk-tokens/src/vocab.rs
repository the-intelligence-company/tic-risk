//! the unified vocabulary: one id space over record kinds, fields (attribute names, figure
//! concepts, state and context fields, economy series), categorical codes, sources, units,
//! neighbour paths and query targets. every entry carries a text (swedish; english where there is
//! one) for the frozen text encoder. ids are positions: specials, the closed tables in table order,
//! then the open entries (series fields, codes) sorted by key. a rebuild with new codes shifts the
//! open ids, so the model maps embeddings by key, never by id.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::fields::Unit;
use crate::texts;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ns {
    Special,
    Kind,
    Field,
    Code,
    Source,
    Unit,
    Path,
    Query,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub id: u32,
    pub ns: Ns,
    pub key: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_en: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

pub const PAD: u32 = 0;
pub const UNKNOWN: u32 = 1;

#[derive(Debug, Default, Clone)]
pub struct Vocab {
    pub entries: Vec<Entry>,
    index: HashMap<(Ns, String), u32>,
}

#[derive(Serialize, Deserialize)]
struct VocabFile {
    version: u32,
    entries: Vec<Entry>,
}

impl Vocab {
    /// specials and every closed table; open entries are added by the build
    pub fn closed() -> Vocab {
        let mut v = Vocab::default();
        v.add(Ns::Special, "pad", "utfyllnad", Some("padding"), None);
        v.add(Ns::Special, "unknown", "okänd", Some("unknown"), None);
        for k in texts::KINDS {
            v.add(Ns::Kind, k.key, k.sv, Some(k.en), None);
        }
        for f in texts::FIELDS {
            v.add(Ns::Field, f.key, f.sv, Some(f.en), Some(f.unit.code()));
        }
        for (key, sv, en) in texts::SOURCES {
            v.add(Ns::Source, key, sv, Some(en), None);
        }
        for (key, sv, en) in texts::UNITS {
            v.add(Ns::Unit, key, sv, Some(en), None);
        }
        for (key, sv, en) in texts::PATHS {
            v.add(Ns::Path, key, sv, Some(en), None);
        }
        for (key, sv, en) in texts::QUERIES {
            v.add(Ns::Query, key, sv, Some(en), None);
        }
        for (field, values) in texts::STATE_CODES {
            for value in values.iter() {
                v.code(field, value);
            }
        }
        for c in risk_store::coverage::SOURCE_COVERAGE {
            v.code("coverage.source", c.source);
        }
        v
    }

    pub fn add(
        &mut self,
        ns: Ns,
        key: &str,
        text: &str,
        text_en: Option<&str>,
        unit: Option<&str>,
    ) -> u32 {
        if let Some(&id) = self.index.get(&(ns, key.to_owned())) {
            return id;
        }
        let id = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        self.entries.push(Entry {
            id,
            ns,
            key: key.to_owned(),
            text: text.to_owned(),
            text_en: text_en.map(str::to_owned),
            unit: unit.map(str::to_owned),
        });
        self.index.insert((ns, key.to_owned()), id);
        id
    }

    pub fn id(&self, ns: Ns, key: &str) -> Option<u32> {
        self.index.get(&(ns, key.to_owned())).copied()
    }

    pub fn kind(&self, key: &str) -> Option<u32> {
        self.id(Ns::Kind, key)
    }

    pub fn field(&self, key: &str) -> Option<u32> {
        self.id(Ns::Field, key)
    }

    pub fn get(&self, id: u32) -> Option<&Entry> {
        self.entries.get(id as usize)
    }

    pub fn key(&self, id: u32) -> &str {
        self.get(id).map_or("?", |e| e.key.as_str())
    }

    /// a categorical value of a field: `field=value`, text from the code tables
    pub fn code(&mut self, field: &str, value: &str) -> u32 {
        let key = format!("{field}={value}");
        if let Some(&id) = self.index.get(&(Ns::Code, key.clone())) {
            return id;
        }
        let (sv, en) = texts::code_text(field, value);
        self.add(Ns::Code, &key, &sv, en.as_deref(), None)
    }

    /// an economy series as a field, text = the series' published name
    pub fn series_field(&mut self, series_id: &str, name: &str, unit: Unit) -> u32 {
        self.add(
            Ns::Field,
            &format!("series.{series_id}"),
            name,
            None,
            Some(unit.code()),
        )
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn count(&self, ns: Ns) -> usize {
        self.entries.iter().filter(|e| e.ns == ns).count()
    }

    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let f = VocabFile {
            version: 1,
            entries: self.entries.clone(),
        };
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), &f)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Vocab, Error> {
        let f: VocabFile = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        let mut v = Vocab::default();
        for e in f.entries {
            if e.id as usize != v.entries.len() {
                return Err(Error::Cache(
                    path.display().to_string(),
                    format!("entry {} out of order", e.key),
                ));
            }
            v.index.insert((e.ns, e.key.clone()), e.id);
            v.entries.push(e);
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn closed_vocab_is_stable_and_textual() {
        let v = Vocab::closed();
        assert_eq!(v.id(Ns::Special, "pad"), Some(PAD));
        assert_eq!(v.id(Ns::Special, "unknown"), Some(UNKNOWN));
        let k = v.kind("bankruptcy.initiated").unwrap();
        assert_eq!(v.get(k).unwrap().text, "Konkurs inledd");
        assert!(v.field("debt.amount").is_some());
        assert!(v.field("figure.net_sales").is_some());
        assert_eq!(
            v.get(v.field("figure.net_sales").unwrap()).unwrap().unit,
            Some("sek".to_owned())
        );
        for e in &v.entries {
            assert!(!e.text.is_empty(), "{} has no text", e.key);
        }
        let n = v.len();
        let mut v2 = Vocab::closed();
        assert_eq!(v2.len(), n, "closed tables are deterministic");
        let a = v2.code("industry.sni", "69101");
        let b = v2.code("industry.sni", "69101");
        assert_eq!(a, b);
        assert!(
            v2.get(a).unwrap().text.contains("Advokat"),
            "SNI text from the code table"
        );
    }
}
