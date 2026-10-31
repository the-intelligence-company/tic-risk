//! the as-of view and the temporal neighbour sampler: the only code that applies `known <= T`.
//! the focal stream under a budget with a per-kind guarantee, neighbours along relations active
//! at T in priority order (parents always), their streams under a smaller budget, the state and
//! context records, the query, and the target segment after T. deterministic for (entity, T,
//! budget).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::context::{Context, EcoView};
use crate::dates::{DAY_MAX, Day, add_months, from_naive, to_naive};
use crate::fields::Unit;
use crate::record::{
    EXTRA_BIT, FLAG_EXTRA, INDEX_BULK_ADDRESS, INDEX_FOCAL_TRUNCATED, INDEX_NEIGHBOURS_TRUNCATED,
    NO_VIA, PeerType, Record, SeqTok, SlotRow,
};
use crate::state::{Cells, ExtraBuf, StateSummary, state_records};
use crate::streams::Streams;
use crate::texts;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Budget {
    /// focal records kept (newest first, every kind keeps its last `per_kind_min`)
    pub focal: usize,
    pub per_kind_min: usize,
    /// records per neighbour
    pub neighbour: usize,
    /// neighbours beyond the parents and owners, which are always included
    pub neighbours_max: usize,
    pub children_max: usize,
    pub owned_max: usize,
    pub cohabitants_max: usize,
    pub siblings_max: usize,
    /// an address with more relations than this is a bulk address: counted, not expanded
    pub address_scan_max: usize,
    pub officer_companies_max: usize,
    /// ended relations older than this are out of view
    pub rel_window_days: i32,
    /// company → person → company; off in the open model, which reads no data about persons
    pub follow_persons: bool,
    /// the target segment: the entity's own records after T within this many months
    pub target_months: u32,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget {
            focal: 1536,
            per_kind_min: 4,
            neighbour: 48,
            neighbours_max: 24,
            children_max: 8,
            owned_max: 8,
            cohabitants_max: 8,
            siblings_max: 6,
            address_scan_max: 5000,
            officer_companies_max: 8,
            rel_window_days: 731,
            follow_persons: false,
            target_months: 24,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Query {
    /// position in `texts::QUERIES`
    pub target: u8,
    pub horizon_months: u32,
}

pub struct Sample {
    /// input tokens: focal (chronological), neighbours by slot, then the extra records
    pub toks: Vec<SeqTok>,
    pub targets: Vec<SeqTok>,
    pub slots: Vec<SlotRow>,
    pub extra: ExtraBuf,
    pub flags: u32,
    pub summary: StateSummary,
    pub neighbours: usize,
}

/// in the as-of view: known and valid by `t`; an ended relation only within the window
pub fn in_view(r: &Record, t: Day, window: i32) -> bool {
    r.known <= t && r.valid <= t && (!r.is_relation() || r.end == DAY_MAX || r.end > t - window)
}

fn active(r: &Record, t: Day) -> bool {
    r.is_relation() && r.known <= t && r.valid <= t && r.end > t
}

/// the entity's records in view, in file order (ascending valid, known)
pub fn eligible(s: &Streams, entity: i32, t: Day, window: i32) -> Vec<u32> {
    let range = s.range(entity);
    range
        .filter(|&i| in_view(&s.records[i], t, window))
        .map(|i| i as u32)
        .collect()
}

/// the newest `budget` records, every kind keeping its last `per_kind_min`; back in valid order
pub fn select(
    s: &Streams,
    mut cand: Vec<u32>,
    budget: usize,
    per_kind_min: usize,
) -> (Vec<u32>, bool) {
    if cand.len() <= budget {
        return (cand, false);
    }
    let key = |i: u32| {
        let r = &s.records[i as usize];
        (r.valid, r.known, i)
    };
    cand.sort_unstable_by_key(|a| std::cmp::Reverse(key(*a)));
    let mut keep = vec![false; cand.len()];
    let mut counts: HashMap<u32, usize> = HashMap::new();
    let mut kept = 0usize;
    for (i, &ri) in cand.iter().enumerate() {
        let c = counts.entry(s.records[ri as usize].kind).or_default();
        if *c < per_kind_min {
            *c += 1;
            keep[i] = true;
            kept += 1;
        }
    }
    for k in keep.iter_mut() {
        if kept >= budget {
            break;
        }
        if !*k {
            *k = true;
            kept += 1;
        }
    }
    // more guaranteed records than the budget: the oldest guaranteed ones go
    for k in keep.iter_mut().rev() {
        if kept <= budget {
            break;
        }
        if *k {
            *k = false;
            kept -= 1;
        }
    }
    let mut out: Vec<u32> = cand
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(r, _)| *r)
        .collect();
    out.sort_unstable_by_key(|&i| key(i));
    (out, true)
}

#[derive(Debug, Clone, Copy)]
struct Neighbour {
    entity: i32,
    hop: u8,
    path: u8,
    via: u32,
    start: Day,
}

struct Found {
    list: Vec<Neighbour>,
    seen: HashSet<i32>,
    flags: u32,
    cohabitants: Option<u32>,
}

impl Found {
    fn add(&mut self, n: Neighbour) -> bool {
        if self.seen.insert(n.entity) {
            self.list.push(n);
            true
        } else {
            false
        }
    }
}

fn newest_first(v: &mut [Neighbour]) {
    v.sort_by(|a, b| b.start.cmp(&a.start).then(a.entity.cmp(&b.entity)));
}

/// the neighbours of an entity at `t` in priority order
fn neighbours(s: &Streams, entity: i32, t: Day, b: &Budget) -> Found {
    let ids = &s.ids;
    let p = &s.paths;
    let mut found = Found {
        list: Vec::new(),
        seen: HashSet::from([entity]),
        flags: 0,
        cohabitants: None,
    };
    let mut parents = Vec::new();
    let mut owners = Vec::new();
    let mut auditors = Vec::new();
    let mut addresses = Vec::new();
    let mut persons = Vec::new();
    for (i, r) in s.range(entity).map(|i| (i as u32, &s.records[i])) {
        if !active(r, t) {
            continue;
        }
        let n = |path: u8, hop: u8| Neighbour {
            entity: i32::try_from(r.peer_id).unwrap_or(0),
            hop,
            path,
            via: i,
            start: r.valid,
        };
        match r.peer_type {
            PeerType::Company if r.kind == ids.group_parent => parents.push(n(p.parent, 1)),
            PeerType::Company if r.kind == ids.ownership_via => owners.push(n(p.owner, 1)),
            PeerType::Company if ids.auditor_kinds.contains(&r.kind) => {
                auditors.push(n(p.auditor, 1))
            }
            PeerType::Address if r.kind == ids.address_registered => addresses.push(r.peer_id),
            PeerType::Person if b.follow_persons && ids.role_kinds.contains(&r.kind) => {
                persons.push(r.peer_id);
            }
            _ => {}
        }
    }
    let reverse = |kind: u32, path: u8| -> Vec<Neighbour> {
        let mut v: Vec<Neighbour> = s
            .rels_to(PeerType::Company, i64::from(entity))
            .iter()
            .map(|&i| (i, &s.records[i as usize]))
            .filter(|(_, r)| r.kind == kind && active(r, t))
            .map(|(i, r)| Neighbour {
                entity: r.entity,
                hop: 1,
                path,
                via: i,
                start: r.valid,
            })
            .collect();
        newest_first(&mut v);
        v
    };
    let mut children = reverse(ids.group_parent, p.child);
    children.truncate(b.children_max);
    let mut owned = reverse(ids.ownership_via, p.owned);
    owned.truncate(b.owned_max);

    let mut cohabitants: Vec<Neighbour> = Vec::new();
    let mut cohabitant_count = 0u32;
    for addr in addresses {
        let list = s.rels_to(PeerType::Address, addr);
        if s.peer_len(PeerType::Address, addr) > b.address_scan_max {
            found.flags |= INDEX_BULK_ADDRESS;
            cohabitant_count = cohabitant_count.max(b.address_scan_max as u32);
            continue;
        }
        for &i in list {
            let r = &s.records[i as usize];
            if r.entity != entity && r.kind == ids.address_registered && active(r, t) {
                cohabitant_count += 1;
                cohabitants.push(Neighbour {
                    entity: r.entity,
                    hop: 1,
                    path: p.cohabitant,
                    via: i,
                    start: r.valid,
                });
            }
        }
    }
    newest_first(&mut cohabitants);
    cohabitants.truncate(b.cohabitants_max);
    found.cohabitants = Some(cohabitant_count);

    let mut hop2: Vec<Neighbour> = Vec::new();
    for parent in &parents {
        for (i, r) in s.range(parent.entity).map(|i| (i as u32, &s.records[i])) {
            if r.kind == ids.group_parent && r.peer_type == PeerType::Company && active(r, t) {
                hop2.push(Neighbour {
                    entity: i32::try_from(r.peer_id).unwrap_or(0),
                    hop: 2,
                    path: p.grandparent,
                    via: i,
                    start: r.valid,
                });
            }
        }
        let mut siblings: Vec<Neighbour> = s
            .rels_to(PeerType::Company, i64::from(parent.entity))
            .iter()
            .map(|&i| (i, &s.records[i as usize]))
            .filter(|(_, r)| r.kind == ids.group_parent && r.entity != entity && active(r, t))
            .map(|(i, r)| Neighbour {
                entity: r.entity,
                hop: 2,
                path: p.sibling,
                via: i,
                start: r.valid,
            })
            .collect();
        newest_first(&mut siblings);
        siblings.truncate(b.siblings_max);
        hop2.extend(siblings);
    }
    let mut via_officer: Vec<Neighbour> = Vec::new();
    for pid in persons {
        for &i in s.rels_to(PeerType::Person, pid) {
            let r = &s.records[i as usize];
            if r.entity != entity && ids.role_kinds.contains(&r.kind) && active(r, t) {
                via_officer.push(Neighbour {
                    entity: r.entity,
                    hop: 2,
                    path: p.via_officer,
                    via: i,
                    start: r.valid,
                });
            }
        }
    }
    newest_first(&mut via_officer);
    via_officer.truncate(b.officer_companies_max);

    for n in parents.into_iter().chain(owners) {
        found.add(n);
    }
    let always = found.list.len();
    let rest = children
        .into_iter()
        .chain(owned)
        .chain(auditors)
        .chain(cohabitants)
        .chain(hop2)
        .chain(via_officer);
    for n in rest {
        if found.list.len() - always >= b.neighbours_max {
            found.flags |= INDEX_NEIGHBOURS_TRUNCATED;
            break;
        }
        found.add(n);
    }
    found
}

/// the neighbours the sample of (entity, T) takes, in slot order: the companies whose streams a
/// caller loading streams per company must read
pub fn neighbour_entities(s: &Streams, entity: i32, t: Day, b: &Budget) -> Vec<i32> {
    neighbours(s, entity, t, b).list.iter().map(|n| n.entity).collect()
}

/// the addresses an entity is registered at by `t`'s relation window, and the parents it has: the
/// peers whose relations the neighbour walk reads (cohabitants, siblings, grandparents)
pub fn walk_peers(s: &Streams, entity: i32) -> (Vec<i64>, Vec<i32>) {
    let ids = &s.ids;
    let mut addresses = Vec::new();
    let mut parents = Vec::new();
    for r in s.records_of(entity) {
        if !r.is_relation() {
            continue;
        }
        match r.peer_type {
            PeerType::Address if r.kind == ids.address_registered => addresses.push(r.peer_id),
            PeerType::Company if r.kind == ids.group_parent => parents.push(i32::try_from(r.peer_id).unwrap_or(0)),
            _ => {}
        }
    }
    addresses.sort_unstable();
    addresses.dedup();
    parents.sort_unstable();
    parents.dedup();
    (addresses, parents)
}

/// the complete sample of one (entity, T): input tokens, slots, extra records, targets
/// a target: the entity's own record after T, inside the target window and before the plan's cut
/// (pretraining stops at the questions' first test as-of, not at the data cut)
fn in_target_window(valid: Day, t: Day, horizon_end: Day, cut: Day) -> bool {
    valid > t && valid <= horizon_end && valid < cut
}

/// `cut`: no target on or after it (the anchor plan's cut; `DAY_MAX` for none)
#[allow(clippy::too_many_arguments)]
pub fn build_sample(
    s: &Streams,
    ctx: &Context,
    eco: &EcoView,
    entity: i32,
    t: Day,
    query: Option<Query>,
    b: &Budget,
    cut: Day,
) -> Sample {
    let focal_all = eligible(s, entity, t, b.rel_window_days);
    let (focal, truncated) = select(s, focal_all.clone(), b.focal, b.per_kind_min);
    let mut flags = if truncated { INDEX_FOCAL_TRUNCATED } else { 0 };
    let mut toks: Vec<SeqTok> = focal
        .iter()
        .map(|&i| SeqTok {
            record: i,
            slot: 0,
            hop: 0,
            path: s.paths.focal,
        })
        .collect();
    let mut slots = vec![SlotRow {
        slot: 0,
        hop: 0,
        path: s.paths.focal,
        entity,
        via: NO_VIA,
    }];
    let found = neighbours(s, entity, t, b);
    flags |= found.flags;
    // a neighbour with no record of its own (only the far end of a relation in the input) gets no
    // slot; skipped here, not in the walk, so a caller loading streams per company still asks for
    // it and gets the sample the full streams give
    for n in found.list.iter().filter(|n| s.entity(n.entity).is_some()) {
        let slot = u16::try_from(slots.len()).unwrap_or(u16::MAX);
        slots.push(SlotRow {
            slot,
            hop: n.hop,
            path: n.path,
            entity: n.entity,
            via: n.via,
        });
        let cand = eligible(s, n.entity, t, b.rel_window_days);
        let (sel, _) = select(s, cand, b.neighbour, 1);
        toks.extend(sel.into_iter().map(|i| SeqTok {
            record: i,
            slot,
            hop: n.hop,
            path: n.path,
        }));
    }

    let mut extra = ExtraBuf::new();
    let summary = state_records(s, ctx, entity, t, &focal_all, found.cohabitants, &mut extra);
    ctx.records(
        s,
        eco,
        entity,
        summary.division,
        summary.section,
        t,
        &mut extra,
    );
    if let Some(q) = query {
        let ids = &s.ids;
        let mut c = Cells::new();
        if let Some((key, _, _)) = texts::QUERIES.get(usize::from(q.target)) {
            c.cat(ids.field("query.target"), ids.code("query.target", key));
        }
        c.months(ids.field("query.horizon_months"), q.horizon_months as f32);
        c.cat(
            ids.field("query.as_of_month"),
            ids.code("query.as_of_month", &{
                use chrono::Datelike;
                to_naive(t).month().to_string()
            }),
        );
        extra.push(
            Record {
                entity,
                kind: ids.query,
                valid: t,
                known: t,
                end: t,
                cell_start: 0,
                cell_count: 0,
                source: ids.source_tic,
                flags: FLAG_EXTRA,
                peer_type: PeerType::None,
                peer_id: 0,
                group: 0,
            },
            &c.cells,
        );
    }
    for i in 0..extra.len() {
        toks.push(SeqTok {
            record: EXTRA_BIT | i as u32,
            slot: 0,
            hop: 0,
            path: s.paths.focal,
        });
    }

    let horizon_end = from_naive(add_months(to_naive(t), b.target_months));
    let targets: Vec<SeqTok> = s
        .range(entity)
        .filter(|&i| in_target_window(s.records[i].valid, t, horizon_end, cut))
        .map(|i| SeqTok {
            record: i as u32,
            slot: 0,
            hop: 0,
            path: s.paths.focal,
        })
        .collect();
    let neighbours = slots.len() - 1;
    let _ = Unit::None;
    Sample {
        toks,
        targets,
        slots,
        extra,
        flags,
        summary,
        neighbours,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::record::FLAG_RELATION;

    #[test]
    fn view_rule() {
        let ev = Record {
            valid: 100,
            known: 105,
            end: 100,
            ..Default::default()
        };
        assert!(!in_view(&ev, 104, 365));
        assert!(in_view(&ev, 105, 365));
        let rel = Record {
            valid: 100,
            known: 100,
            end: 200,
            flags: FLAG_RELATION,
            ..Default::default()
        };
        assert!(in_view(&rel, 150, 365));
        assert!(in_view(&rel, 300, 365), "ended inside the window");
        assert!(!in_view(&rel, 600, 365), "ended before the window");
        assert!(active(&rel, 150));
        assert!(!active(&rel, 200));
        let open = Record {
            end: DAY_MAX,
            ..rel
        };
        assert!(in_view(&open, 5000, 365));
    }

    #[test]
    fn targets_stop_at_the_cut() {
        let (t, end, cut) = (1000, 1730, 1200);
        assert!(!in_target_window(1000, t, end, cut), "T itself is input, not target");
        assert!(in_target_window(1001, t, end, cut));
        assert!(in_target_window(1199, t, end, cut));
        assert!(!in_target_window(1200, t, end, cut), "the cut day is outside");
        assert!(!in_target_window(1500, t, end, cut), "inside the window, past the cut");
        assert!(in_target_window(1730, t, end, DAY_MAX), "no cut: the window's end is inside");
        assert!(!in_target_window(1731, t, end, DAY_MAX));
    }

    #[test]
    fn a_neighbour_without_records_gets_no_slot() {
        use crate::build::MemBuild;
        use crate::context::Context;
        use crate::vocab::Vocab;
        use risk_eval::store::{Economy, Sector};
        let mut m = MemBuild::new(Vocab::closed());
        m.events([
            r#"{"c":1,"k":"bankruptcy.initiated","t":"2020-06-01","ka":"2020-06-01","est":false,"s":"bolagsverket","a":{}}"#,
            r#"{"c":2,"k":"bankruptcy.initiated","t":"2020-06-01","ka":"2020-06-01","est":false,"s":"bolagsverket","a":{}}"#,
        ])
        .unwrap();
        // 3 is only the far end of a relation
        m.relations([
            r#"{"a":{"t":"company","id":1},"k":"ownership.beneficial_via","b":{"t":"company","id":2},"from":"2019-01-01","to":null,"ka":"2019-01-01","est":false,"x":{}}"#,
            r#"{"a":{"t":"company","id":1},"k":"ownership.beneficial_via","b":{"t":"company","id":3},"from":"2019-01-01","to":null,"ka":"2019-01-01","est":false,"x":{}}"#,
        ])
        .unwrap();
        let (s, _) = m.finish(&[]).unwrap();
        assert!(s.entity(2).is_some() && s.entity(3).is_none());
        let ctx = Context::from_parts(
            Economy { national: vec![], industry: vec![] },
            Sector::from_lines(std::iter::empty(), None).unwrap(),
            &s.vocab,
            vec![],
        )
        .unwrap();
        let t = crate::dates::from_naive(chrono::NaiveDate::from_ymd_opt(2021, 1, 1).unwrap());
        let b = Budget::default();
        let walk = neighbour_entities(&s, 1, t, &b);
        assert!(walk.contains(&2) && walk.contains(&3), "the walk lists both, so a fetch of the neighbours asks for both");
        let sample = build_sample(&s, &ctx, &ctx.view(t), 1, t, None, &b, DAY_MAX);
        let slots: Vec<(u16, i32)> = sample.slots.iter().map(|x| (x.slot, x.entity)).collect();
        assert_eq!(slots, vec![(0, 1), (1, 2)], "no slot for 3, the slots contiguous");
        assert_eq!(sample.neighbours, 1);
        let focal_peers: Vec<u64> = sample
            .toks
            .iter()
            .filter(|x| x.slot == 0 && x.record & EXTRA_BIT == 0)
            .map(|x| &s.records[x.record as usize])
            .filter(|r| r.is_relation())
            .map(|r| r.peer_id as u64)
            .collect();
        assert!(focal_peers.contains(&3), "the relation to 3 stays a token of the focal stream");
    }
}
