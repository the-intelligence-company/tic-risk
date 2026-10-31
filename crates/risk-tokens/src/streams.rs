//! the loaded token cache: records and cells in memory, the entity index, the reverse index of
//! relations by peer (parent → children, address → companies, person → companies), and the
//! vocabulary ids the state derivation and the sampler need, resolved once. read-only after open;
//! shared across the sampling threads.

use std::collections::HashMap;
use std::path::Path;

use crate::Error;
use crate::record::{
    CELLS_MAGIC, Cell, CellType, ENTITIES_MAGIC, EntityRow, PeerType, RECORDS_MAGIC, Record,
    read_all,
};
use crate::texts;
use crate::vocab::{Ns, Vocab};

/// positions in `texts::PATHS`
#[derive(Debug, Clone, Copy)]
pub struct PathIds {
    pub focal: u8,
    pub parent: u8,
    pub grandparent: u8,
    pub child: u8,
    pub sibling: u8,
    pub owner: u8,
    pub owned: u8,
    pub cohabitant: u8,
    pub auditor: u8,
    pub via_officer: u8,
}

fn path(key: &str) -> u8 {
    texts::PATHS
        .iter()
        .position(|(k, _, _)| *k == key)
        .and_then(|i| u8::try_from(i).ok())
        .unwrap_or(0)
}

/// vocabulary ids the sampler and the state derivation read; a missing closed entry is a build
/// error, so these are plain ids
#[derive(Debug, Clone)]
pub struct Ids {
    pub group_parent: u32,
    pub ownership_via: u32,
    pub address_registered: u32,
    pub auditor_kinds: Vec<u32>,
    pub role_kinds: Vec<u32>,
    pub industry_code: u32,
    pub report_figures: u32,
    pub report_received: u32,
    pub report_registered: u32,
    pub report_approved: u32,
    pub board_gap_started: u32,
    pub board_gap_ended: u32,
    pub debt_registered: u32,
    /// a debt present from the start of the record (absent from some vocabularies)
    pub debt_present: Option<u32>,
    pub debt_changed: u32,
    pub debt_cleared: u32,
    pub tax_registration: u32,
    pub tax_deferral: u32,
    pub liquidation_decided: u32,
    pub liquidation_ended: u32,
    pub reconstruction_decided: u32,
    pub reconstruction_ended: u32,
    pub liquidation_order_issued: u32,
    pub liquidation_order_recalled: u32,
    pub permit_granted: u32,
    pub permit_ended: u32,
    pub case_received: u32,
    pub case_terminated: u32,
    pub state_profile: u32,
    pub state_interval: u32,
    pub state_filing: u32,
    pub state_sector: u32,
    pub state_coverage: u32,
    pub context_economy: u32,
    pub context_industry: u32,
    pub query: u32,
    pub source_tic: u8,
    pub fields: HashMap<&'static str, u32>,
    pub codes: HashMap<String, u32>,
}

impl Ids {
    fn resolve(v: &Vocab) -> Result<Ids, Error> {
        let kind = |k: &str| {
            v.kind(k)
                .ok_or_else(|| Error::Other(format!("vocabulary lacks kind {k}")))
        };
        let mut fields = HashMap::new();
        for f in texts::FIELDS {
            if let Some(id) = v.field(f.key) {
                fields.insert(f.key, id);
            }
        }
        let mut codes = HashMap::new();
        for e in &v.entries {
            if e.ns == Ns::Code {
                codes.insert(e.key.clone(), e.id);
            }
        }
        let role_kinds: Vec<u32> = v
            .entries
            .iter()
            .filter(|e| e.ns == Ns::Kind && e.key.starts_with("role."))
            .map(|e| e.id)
            .collect();
        Ok(Ids {
            group_parent: kind("group.parent")?,
            ownership_via: kind("ownership.beneficial_via")?,
            address_registered: kind("address.registered")?,
            auditor_kinds: vec![
                kind("role.auditor")?,
                kind("role.auditor_in_charge")?,
                kind("role.deputy_auditor")?,
            ],
            role_kinds,
            industry_code: kind("industry.code")?,
            report_figures: kind("report.figures")?,
            report_received: kind("report.received")?,
            report_registered: kind("report.registered")?,
            report_approved: kind("report.approved")?,
            board_gap_started: kind("board.gap_started")?,
            board_gap_ended: kind("board.gap_ended")?,
            debt_registered: kind("debt.registered")?,
            debt_present: v.kind("debt.present"),
            debt_changed: kind("debt.changed")?,
            debt_cleared: kind("debt.cleared")?,
            tax_registration: kind("tax.registration")?,
            tax_deferral: kind("tax.deferral")?,
            liquidation_decided: kind("liquidation.decided")?,
            liquidation_ended: kind("liquidation.ended")?,
            reconstruction_decided: kind("reconstruction.decided")?,
            reconstruction_ended: kind("reconstruction.ended")?,
            liquidation_order_issued: kind("liquidation_order.issued")?,
            liquidation_order_recalled: kind("liquidation_order.recalled")?,
            permit_granted: kind("permit.serving_granted")?,
            permit_ended: kind("permit.serving_ended")?,
            case_received: kind("case.received")?,
            case_terminated: kind("case.terminated")?,
            state_profile: kind("state.profile")?,
            state_interval: kind("state.interval")?,
            state_filing: kind("state.filing")?,
            state_sector: kind("state.sector")?,
            state_coverage: kind("state.coverage")?,
            context_economy: kind("context.economy")?,
            context_industry: kind("context.industry")?,
            query: kind("query")?,
            source_tic: texts::source_index("tic"),
            fields,
            codes,
        })
    }

    pub fn field(&self, key: &str) -> u32 {
        self.fields
            .get(key)
            .copied()
            .unwrap_or(crate::vocab::UNKNOWN)
    }

    /// a code id by `field=value`; `UNKNOWN` when the value was never seen
    pub fn code(&self, field: &str, value: &str) -> u32 {
        self.codes
            .get(&format!("{field}={value}"))
            .copied()
            .unwrap_or(crate::vocab::UNKNOWN)
    }
}

pub struct Streams {
    pub vocab: Vocab,
    pub records: Vec<Record>,
    pub cells: Vec<Cell>,
    pub entities: Vec<EntityRow>,
    ent_index: HashMap<i32, u32>,
    by_peer: HashMap<(u8, i64), Vec<u32>>,
    /// the number of relations at a peer when the streams hold fewer than all of them: a bulk
    /// address whose relations were counted instead of read (the sampler only asks whether it is
    /// above its scan limit)
    peer_counts: HashMap<(u8, i64), usize>,
    pub ids: Ids,
    pub paths: PathIds,
}

impl Streams {
    pub fn open(tokens_dir: &Path) -> Result<Streams, Error> {
        let vocab = Vocab::load(&tokens_dir.join("vocab.json"))?;
        let records: Vec<Record> = read_all(&tokens_dir.join("records.bin"), RECORDS_MAGIC)?;
        let cells: Vec<Cell> = read_all(&tokens_dir.join("cells.bin"), CELLS_MAGIC)?;
        let entities: Vec<EntityRow> = read_all(&tokens_dir.join("entities.bin"), ENTITIES_MAGIC)?;
        tracing::info!(
            records = records.len(),
            cells = cells.len(),
            entities = entities.len(),
            vocab = vocab.len(),
            "token cache loaded"
        );
        Streams::from_parts(vocab, records, cells, entities)
    }

    /// streams over records sorted by (entity, valid, known, kind), their cells and the entity index
    pub fn from_parts(vocab: Vocab, records: Vec<Record>, cells: Vec<Cell>, entities: Vec<EntityRow>) -> Result<Streams, Error> {
        let ids = Ids::resolve(&vocab)?;
        let ent_index: HashMap<i32, u32> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.entity, i as u32))
            .collect();
        let mut by_peer: HashMap<(u8, i64), Vec<u32>> = HashMap::new();
        for (i, r) in records.iter().enumerate() {
            if r.is_relation() && r.peer_type != PeerType::None {
                by_peer
                    .entry((r.peer_type as u8, r.peer_id))
                    .or_default()
                    .push(i as u32);
            }
        }
        let paths = PathIds {
            focal: path("focal"),
            parent: path("parent"),
            grandparent: path("grandparent"),
            child: path("child"),
            sibling: path("sibling"),
            owner: path("owner"),
            owned: path("owned"),
            cohabitant: path("cohabitant"),
            auditor: path("auditor"),
            via_officer: path("via_officer"),
        };
        Ok(Streams {
            vocab,
            records,
            cells,
            entities,
            ent_index,
            by_peer,
            peer_counts: HashMap::new(),
            ids,
            paths,
        })
    }

    /// how many relations point at a peer: the counted number where one was set, else the
    /// relations held
    pub fn peer_len(&self, peer_type: PeerType, id: i64) -> usize {
        match self.peer_counts.get(&(peer_type as u8, id)) {
            Some(&n) => n,
            None => self.rels_to(peer_type, id).len(),
        }
    }

    /// the number of relations at a peer the streams hold only part of (a bulk address)
    pub fn set_peer_count(&mut self, peer_type: PeerType, id: i64, n: usize) {
        self.peer_counts.insert((peer_type as u8, id), n);
    }

    pub fn entity(&self, id: i32) -> Option<&EntityRow> {
        self.ent_index.get(&id).map(|&i| &self.entities[i as usize])
    }

    /// the record index range of an entity (empty when unknown)
    pub fn range(&self, id: i32) -> std::ops::Range<usize> {
        match self.entity(id) {
            Some(e) => e.rec_start as usize..e.rec_start as usize + e.rec_count as usize,
            None => 0..0,
        }
    }

    pub fn records_of(&self, id: i32) -> &[Record] {
        &self.records[self.range(id)]
    }

    pub fn cells_of(&self, r: &Record) -> &[Cell] {
        self.cells.get(r.cells()).unwrap_or(&[])
    }

    /// relation records (indices) whose peer is this endpoint
    pub fn rels_to(&self, peer_type: PeerType, id: i64) -> &[u32] {
        self.by_peer
            .get(&(peer_type as u8, id))
            .map_or(&[], Vec::as_slice)
    }

    pub fn cell_value(&self, r: &Record, field: u32) -> Option<f32> {
        self.cells_of(r)
            .iter()
            .find(|c| c.field == field && c.ctype != CellType::Cat)
            .map(|c| c.value)
    }

    pub fn cell_code(&self, r: &Record, field: u32) -> Option<u32> {
        self.cells_of(r)
            .iter()
            .find(|c| c.field == field && c.ctype == CellType::Cat)
            .map(|c| c.code)
    }

    /// the value part of a code's key (`industry.sni=69101` → `69101`)
    pub fn code_value(&self, code: u32) -> &str {
        self.vocab
            .get(code)
            .and_then(|e| e.key.split_once('='))
            .map_or("", |(_, v)| v)
    }

    pub fn kind_key(&self, kind: u32) -> &str {
        self.vocab.key(kind)
    }
}
