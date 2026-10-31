//! the input from files: one company's facts as a directory of lines, read and built into the
//! view the model reads (`view.rs`). the lines are read in file order, which is the build's tie
//! order

use std::fs;
use std::path::Path;

use chrono::NaiveDate;
use risk_eval::store::{Economy, Sector, read_lives};
use risk_model::vocab::ModelDir;
use risk_store::population::CompanyLife;
use risk_tokens::build::BuildStats;
use risk_tokens::vocab::Vocab;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::view::{FetchStats, Fetched, InputLines, View, build_view, life_line};

pub const FORMAT: &str = "tic-risk-input/1";

const FILES: [&str; 6] = ["events.jsonl", "relations.jsonl", "economy.jsonl", "sector.jsonl", "calendar.jsonl", "lives.csv"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub format: String,
    /// the company the input is for
    pub focal: i32,
    /// the time the data is from (RFC 3339): nothing known later is in the input
    pub read_at: String,
    /// the as-of days the input was prepared for: the related companies the model reads on these
    /// days are in it; another day may want one it does not hold
    pub as_of: Vec<NaiveDate>,
    /// companies, persons, addresses, case and report numbers are pseudonyms
    pub pseudonyms: bool,
    /// what to call the company in a report (an example's name), when the input names it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

pub struct Input {
    pub meta: Meta,
    pub lives: Vec<CompanyLife>,
    pub events: Vec<String>,
    pub relations: Vec<String>,
    pub economy: Vec<String>,
    pub sector: Vec<String>,
    pub calendar: Vec<String>,
}

fn lines(path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(path).map_err(|e| Error::Other(format!("{}: {e}", path.display())))?;
    Ok(text.lines().filter(|l| !l.trim().is_empty()).map(str::to_owned).collect())
}

fn write_lines(path: &Path, lines: &[String]) -> Result<()> {
    let mut text = String::with_capacity(lines.iter().map(|l| l.len() + 1).sum());
    for l in lines {
        text.push_str(l);
        text.push('\n');
    }
    fs::write(path, text).map_err(|e| Error::Other(format!("{}: {e}", path.display())))
}

impl Input {
    pub fn read(dir: &Path) -> Result<Input> {
        let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).map_err(|e| Error::Other(format!("{}: {e}", dir.join("meta.json").display())))?)?;
        if meta.format != FORMAT {
            return Err(Error::Other(format!("{}: format {} is not {FORMAT}", dir.display(), meta.format)));
        }
        Ok(Input {
            lives: read_lives(&dir.join("lives.csv")).map_err(|e| Error::Other(format!("lives.csv: {e}")))?,
            events: lines(&dir.join("events.jsonl"))?,
            relations: lines(&dir.join("relations.jsonl"))?,
            economy: lines(&dir.join("economy.jsonl"))?,
            sector: lines(&dir.join("sector.jsonl"))?,
            calendar: lines(&dir.join("calendar.jsonl"))?,
            meta,
        })
    }

    /// the files, into `dir` (created); other files there are left alone
    pub fn write(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&self.meta)? + "\n")?;
        let mut lives = vec!["company_id,legal_form,registered,ceased".to_owned()];
        lives.extend(self.lives.iter().map(|l| {
            let day = |d: Option<NaiveDate>| d.map(|d| d.to_string()).unwrap_or_default();
            format!("{},{},{},{}", l.company_id, l.legal_form.code(), day(l.registered), day(l.ceased))
        }));
        let parts: [&[String]; 6] = [&self.events, &self.relations, &self.economy, &self.sector, &self.calendar, &lives];
        for (name, part) in FILES.iter().zip(parts) {
            write_lines(&dir.join(name), part)?;
        }
        Ok(())
    }

    /// the context tables the input carries (the economy, the industry key figures), parsed once:
    /// an explanation rebuilds the view many times over the same tables
    pub fn context(&self) -> Result<(Economy, Sector)> {
        let economy = Economy::from_lines(self.economy.iter().map(String::as_str)).map_err(|e| Error::Other(format!("economy.jsonl: {e}")))?;
        let calendar: Vec<&str> = self.calendar.iter().map(String::as_str).collect();
        let sector = Sector::from_lines(self.sector.iter().map(String::as_str), Some(&calendar)).map_err(|e| Error::Other(format!("sector.jsonl: {e}")))?;
        Ok((economy, sector))
    }

    /// the view the model reads, built from the files
    pub fn fetched(&self, corpus: &Vocab, md: &ModelDir) -> Result<Fetched> {
        let (economy, sector) = self.context()?;
        Ok(self.build(corpus, md, economy, sector)?.0)
    }

    /// the view over the given context tables, with the build's own account of what it read and
    /// what it left out (`tic-risk check`)
    pub fn build(&self, corpus: &Vocab, md: &ModelDir, economy: Economy, sector: Sector) -> Result<(Fetched, BuildStats)> {
        let View { streams, ctx, remap, bstats, new_codes, unknown_keys } = build_view(corpus, md, &self.events, &self.relations, &self.lives, economy, sector)?;
        let stats = FetchStats {
            read_at: self.meta.read_at.clone(),
            companies: self.lives.len(),
            events: bstats.events,
            relations: bstats.relations,
            records: bstats.records,
            cells: bstats.cells,
            new_codes,
            unknown_keys,
            ..FetchStats::default()
        };
        let fetched = Fetched {
            focal: self.meta.focal,
            streams,
            ctx,
            remap,
            stats,
            lines: InputLines { events: self.events.clone(), relations: self.relations.clone(), lives: self.lives.iter().map(life_line).collect() },
            lives: self.lives.clone(),
        };
        Ok((fetched, bstats))
    }
}
