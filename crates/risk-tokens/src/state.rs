//! the as-of state tokens: profile, active intervals, the latest filing with the fiscal calendar
//! and the overdue rule, the comparison with the industry quartiles, and the source coverage,
//! derived at T from the focal records in view. written as extra records of a sample (source
//! `tic`), never stored as truth. the figures read are the canonical set: one per company and
//! period, merged across the report sources.

use std::collections::{BTreeSet, HashMap, HashSet};

use chrono::Datelike;
use risk_eval::store::{FIGURES, band_for};

use crate::context::Context;
use crate::dates::{DAY_MAX, DAY_MIN, Day, add_months, from_naive, months_between, to_naive};
use crate::fields::Unit;
use crate::record::{Cell, CellType, FLAG_EXTRA, PeerType, Record};
use crate::streams::Streams;
use crate::vocab::UNKNOWN;

/// the share of untaxed reserves counted as equity (one minus the corporate tax rate)
pub const UNTAXED_EQUITY_SHARE: f32 = 0.794;
/// AGM within six months of the period end, filing within a month after: the statutory latest
pub const FILING_DEADLINE_MONTHS: u32 = 7;
/// a first fiscal year may run 18 months; its report is due 7 months later
pub const FIRST_REPORT_MONTHS: u32 = 25;
/// a report whose period end lies this close to an expected period end covers it
const PERIOD_MATCH_DAYS: i32 = 45;
/// sector figure (`risk_eval::store::FIGURES` order) → own figure and the factor to the
/// statistic's unit; sales per employee is computed
const SECTOR_MAP: [(Option<&str>, f32); 8] = [
    (Some("figure.debt_equity"), 1.0),
    (Some("figure.equity_ratio"), 100.0),
    (Some("figure.net_margin"), 100.0),
    (None, 1.0),
    (Some("figure.operating_margin"), 100.0),
    (Some("figure.quick_ratio"), 100.0),
    (Some("figure.roa"), 100.0),
    (Some("figure.roe"), 100.0),
];
const BAND_LABELS: [&str; 10] = [
    "all", "0", "1-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200-499", "500+",
];

#[derive(Default)]
pub struct Cells {
    pub cells: Vec<Cell>,
}

impl Cells {
    pub fn new() -> Cells {
        Cells::default()
    }

    pub fn push(&mut self, field: u32, ctype: CellType, unit: Unit, value: f32) {
        if field == UNKNOWN || !value.is_finite() {
            return;
        }
        self.cells.push(Cell {
            field,
            code: 0,
            value,
            ctype,
            unit: unit as u8,
        });
    }

    pub fn num(&mut self, field: u32, unit: Unit, value: f32) {
        self.push(field, CellType::Num, unit, value);
    }

    pub fn cat(&mut self, field: u32, code: u32) {
        if field == UNKNOWN || code == UNKNOWN {
            return;
        }
        self.cells.push(Cell {
            field,
            code,
            value: 0.0,
            ctype: CellType::Cat,
            unit: Unit::None as u8,
        });
    }

    pub fn flag(&mut self, field: u32, on: bool) {
        self.push(
            field,
            CellType::Bool,
            Unit::None,
            if on { 1.0 } else { 0.0 },
        );
    }

    pub fn days(&mut self, field: u32, value: f32) {
        self.push(field, CellType::Days, Unit::Days, value);
    }

    pub fn months(&mut self, field: u32, value: f32) {
        self.push(field, CellType::Months, Unit::Months, value);
    }
}

/// the state, context and query records of one sample, with their cells
#[derive(Default)]
pub struct ExtraBuf {
    pub records: Vec<Record>,
    pub cells: Vec<Cell>,
}

impl ExtraBuf {
    pub fn new() -> ExtraBuf {
        ExtraBuf::default()
    }

    pub fn push(&mut self, mut rec: Record, cells: &[Cell]) {
        rec.cell_start = u32::try_from(self.cells.len()).unwrap_or(u32::MAX);
        rec.cell_count = u16::try_from(cells.len()).unwrap_or(u16::MAX);
        rec.flags |= FLAG_EXTRA;
        self.cells.extend_from_slice(cells);
        self.records.push(rec);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[derive(Debug, Default, Clone)]
pub struct StateSummary {
    pub sni: Option<String>,
    pub division: Option<u32>,
    pub section: Option<char>,
    pub employees: Option<f32>,
    pub fye_month: Option<u32>,
    pub age_months: Option<i32>,
    pub reports_known: u32,
    pub overdue: bool,
    pub sector_records: u16,
}

pub fn size_band(employees: Option<f32>) -> &'static str {
    let Some(e) = employees.filter(|e| e.is_finite()) else {
        return "unknown";
    };
    BAND_LABELS[usize::from(band_for(Some(e))).clamp(1, 9)]
}

fn ratio(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(a), Some(b)) if b != 0.0 && a.is_finite() && b.is_finite() => Some(a / b),
        _ => None,
    }
}

/// ((case type code, debt type code), (amount, cases, since))
/// ((case type, debt type), (amount, cases, since)); since is None for a debt present from the
/// start of the record
type DebtState = ((u32, u32), (f32, f32, Option<Day>));

struct Scan {
    events: u32,
    relations: u32,
    /// (code id of `industry.sni=…`, value, rank, inferred): the company's industry, the registered
    /// code once known, before it the code inferred from the purpose text
    sni: Option<(u32, String, f32, bool)>,
    gaps: HashMap<u32, Day>,
    debts: HashMap<(u32, u32), (f32, f32, Option<Day>)>,
    tax: Option<(u32, bool, bool, Day)>,
    deferral: Option<(f32, Day)>,
    liquidation: Option<Day>,
    reconstruction: Option<Day>,
    order: Option<(Day, u32)>,
    permits: i32,
    permit_since: Option<Day>,
    open_cases: HashSet<u32>,
    period_ends: BTreeSet<Day>,
    figures: Vec<u32>,
}

fn scan(s: &Streams, view: &[u32]) -> Scan {
    let ids = &s.ids;
    let f = |k: &str| ids.field(k);
    let (f_sni, f_rank, f_gap, f_case_type, f_debt_type, f_amount, f_cases) = (
        f("industry.sni"),
        f("industry.rank"),
        f("board.gap"),
        f("debt.case_type"),
        f("debt.debt_type"),
        f("debt.amount"),
        f("debt.cases"),
    );
    let (f_fskatt, f_vat, f_employer, f_outstanding, f_period_end, f_cause) = (
        f("tax.fskatt"),
        f("tax.vat"),
        f("tax.employer"),
        f("tax.outstanding"),
        f("report.period_end"),
        f("liquidation_order.cause"),
    );
    let mut sc = Scan {
        events: 0,
        relations: 0,
        sni: None,
        gaps: HashMap::new(),
        debts: HashMap::new(),
        tax: None,
        deferral: None,
        liquidation: None,
        reconstruction: None,
        order: None,
        permits: 0,
        permit_since: None,
        open_cases: HashSet::new(),
        period_ends: BTreeSet::new(),
        figures: Vec::new(),
    };
    for &i in view {
        let r = &s.records[i as usize];
        if r.is_relation() {
            sc.relations += 1;
            continue;
        }
        sc.events += 1;
        let k = r.kind;
        if k == ids.industry_code {
            if let Some(code) = s.cell_code(r, f_sni) {
                let rank = s.cell_value(r, f_rank).unwrap_or(1.0);
                // the latest rank-1 code wins; a lower-ranked code only fills an empty slot
                // the registered code over the inferred one, then the lower rank; among equals the
                // latest wins. the classifier's code is the fallback: it never replaces a registered one
                let inferred = r.source == ids.source_tic;
                let take = match &sc.sni {
                    None => true,
                    Some((_, _, prev_rank, prev_inferred)) => (inferred, rank) <= (*prev_inferred, *prev_rank),
                };
                if take {
                    sc.sni = Some((code, s.code_value(code).to_owned(), rank, inferred));
                }
            }
        } else if k == ids.board_gap_started {
            if let Some(g) = s.cell_code(r, f_gap) {
                sc.gaps.entry(g).or_insert(r.valid);
            }
        } else if k == ids.board_gap_ended {
            if let Some(g) = s.cell_code(r, f_gap) {
                sc.gaps.remove(&g);
            }
        } else if k == ids.debt_registered || k == ids.debt_changed || Some(k) == ids.debt_present {
            let key = (
                s.cell_code(r, f_case_type).unwrap_or(0),
                s.cell_code(r, f_debt_type).unwrap_or(0),
            );
            let amount = s.cell_value(r, f_amount).unwrap_or(0.0);
            let cases = s.cell_value(r, f_cases).unwrap_or(0.0);
            let since = match sc.debts.get(&key) {
                Some(d) => d.2,
                None if Some(k) == ids.debt_present => None,
                None => Some(r.valid),
            };
            sc.debts.insert(key, (amount, cases, since));
        } else if k == ids.debt_cleared {
            let key = (
                s.cell_code(r, f_case_type).unwrap_or(0),
                s.cell_code(r, f_debt_type).unwrap_or(0),
            );
            sc.debts.remove(&key);
        } else if k == ids.tax_registration {
            sc.tax = Some((
                s.cell_code(r, f_fskatt).unwrap_or(UNKNOWN),
                s.cell_value(r, f_vat).unwrap_or(0.0) > 0.5,
                s.cell_value(r, f_employer).unwrap_or(0.0) > 0.5,
                r.valid,
            ));
        } else if k == ids.tax_deferral {
            let outstanding = s.cell_value(r, f_outstanding).unwrap_or(0.0);
            sc.deferral = if outstanding > 0.0 {
                Some((outstanding, sc.deferral.map_or(r.valid, |d| d.1)))
            } else {
                None
            };
        } else if k == ids.liquidation_decided {
            sc.liquidation = Some(r.valid);
        } else if k == ids.liquidation_ended {
            sc.liquidation = None;
        } else if k == ids.reconstruction_decided {
            sc.reconstruction = Some(r.valid);
        } else if k == ids.reconstruction_ended {
            sc.reconstruction = None;
        } else if k == ids.liquidation_order_issued {
            sc.order = Some((r.valid, s.cell_code(r, f_cause).unwrap_or(UNKNOWN)));
        } else if k == ids.liquidation_order_recalled {
            sc.order = None;
        } else if k == ids.permit_granted {
            sc.permits += 1;
            sc.permit_since.get_or_insert(r.valid);
        } else if k == ids.permit_ended {
            sc.permits -= 1;
            if sc.permits <= 0 {
                sc.permits = 0;
                sc.permit_since = None;
            }
        } else if k == ids.case_received {
            if r.group != 0 {
                sc.open_cases.insert(r.group);
            }
        } else if k == ids.case_terminated {
            sc.open_cases.remove(&r.group);
        } else if k == ids.report_received || k == ids.report_registered || k == ids.report_approved
        {
            if let Some(v) = s.cell_value(r, f_period_end) {
                sc.period_ends.insert(r.valid - v as Day);
            }
        } else if k == ids.report_figures {
            sc.figures.push(i);
            sc.period_ends.insert(r.valid);
        }
    }
    sc
}

/// the state records of one entity at `t` from its records in view (ascending valid)
/// the coverage source the filing state depends on
pub const REPORT_SOURCE: &str = "annual_report_registrations";

pub fn state_records(
    s: &Streams,
    ctx: &Context,
    entity: i32,
    t: Day,
    view: &[u32],
    cohabitants: Option<u32>,
    out: &mut ExtraBuf,
) -> StateSummary {
    let ids = &s.ids;
    let f = |k: &str| ids.field(k);
    let code = |field: &str, value: &str| ids.code(field, value);
    let base = Record {
        entity,
        kind: ids.state_profile,
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
    };
    let sc = scan(s, view);
    let mut summary = StateSummary::default();

    // latest filing
    let latest = sc.figures.last().map(|&i| &s.records[i as usize]);
    let fig = |r: &Record, name: &str| s.cell_value(r, f(name));
    summary.employees = latest.and_then(|r| fig(r, "figure.employees"));
    summary.sni = sc.sni.as_ref().map(|(_, v, _, _)| v.clone());
    summary.division = summary
        .sni
        .as_ref()
        .and_then(|v| v.get(..2))
        .and_then(|d| d.parse().ok());
    summary.section = summary.division.and_then(risk_eval::store::section_of);
    let ent = s.entity(entity);
    let registered = ent.map(|e| e.registered).filter(|r| *r != DAY_MIN);
    summary.age_months = registered.map(|r| months_between(r, t));

    // profile
    let mut c = Cells::new();
    if let Some(e) = ent
        && let Some(lf) = ["AB", "HB", "KB", "EF"].get(usize::from(e.legal_form))
    {
        c.cat(f("profile.legal_form"), code("profile.legal_form", lf));
    }
    if let Some(a) = summary.age_months {
        c.months(f("profile.age_months"), a as f32);
    }
    if let Some((sni_code, _, _, _)) = &sc.sni {
        c.cat(f("profile.sni"), *sni_code);
    }
    if let Some(sec) = summary.section {
        c.cat(
            f("profile.section"),
            code("profile.section", &sec.to_string()),
        );
    }
    c.cat(
        f("profile.size_band"),
        code("profile.size_band", size_band(summary.employees)),
    );
    c.num(f("profile.events_known"), Unit::Count, sc.events as f32);
    c.num(
        f("profile.relations_known"),
        Unit::Count,
        sc.relations as f32,
    );
    if let Some(n) = cohabitants {
        c.num(f("profile.address_cohabitants"), Unit::Count, n as f32);
    }
    out.push(base, &c.cells);

    // intervals
    let interval =
        |out: &mut ExtraBuf, name: &str, since: Option<Day>, extra: &dyn Fn(&mut Cells)| {
            let mut c = Cells::new();
            c.cat(f("interval.name"), code("interval.name", name));
            if let Some(d) = since {
                c.days(f("interval.days"), (t - d) as f32);
            }
            extra(&mut c);
            out.push(
                Record {
                    kind: ids.state_interval,
                    ..base
                },
                &c.cells,
            );
        };
    let mut gaps: Vec<(u32, Day)> = sc.gaps.iter().map(|(k, v)| (*k, *v)).collect();
    gaps.sort_unstable();
    for (g, since) in gaps {
        interval(out, "board_gap", Some(since), &|c| {
            c.cat(f("interval.code"), g)
        });
    }
    let mut debts: Vec<DebtState> = sc.debts.iter().map(|(k, v)| (*k, *v)).collect();
    debts.sort_by_key(|a| a.0);
    for ((_, debt_type), (amount, cases, since)) in debts {
        interval(out, "kfm_debt", since, &|c| {
            c.cat(f("interval.code"), debt_type);
            c.num(f("interval.amount"), Unit::Sek, amount);
            c.num(f("interval.cases"), Unit::Count, cases);
        });
    }
    if let Some((fskatt, vat, employer, since)) = sc.tax {
        interval(out, "fskatt", Some(since), &|c| {
            c.cat(f("interval.code"), fskatt)
        });
        if vat {
            interval(out, "vat", Some(since), &|_| {});
        }
        if employer {
            interval(out, "employer", Some(since), &|_| {});
        }
    }
    if let Some((outstanding, since)) = sc.deferral {
        interval(out, "tax_deferral", Some(since), &|c| {
            c.num(f("interval.amount"), Unit::Sek, outstanding);
        });
    }
    if let Some(d) = sc.liquidation {
        interval(out, "liquidation", Some(d), &|_| {});
    }
    if let Some(d) = sc.reconstruction {
        interval(out, "reconstruction", Some(d), &|_| {});
    }
    if let Some((d, cause)) = sc.order {
        interval(out, "liquidation_order", Some(d), &|c| {
            c.cat(f("interval.code"), cause)
        });
    }
    if sc.permits > 0 {
        interval(out, "serving_permit", sc.permit_since, &|c| {
            c.num(f("interval.cases"), Unit::Count, sc.permits as f32);
        });
    }
    if !sc.open_cases.is_empty() {
        interval(out, "open_cases", None, &|c| {
            c.num(f("interval.cases"), Unit::Count, sc.open_cases.len() as f32);
        });
    }

    // filing and the fiscal calendar. the filing state needs the report family: before its
    // source is live at T, no report in view says nothing (it would read "overdue" for that
    // reason alone), so the state is left out and the coverage record says why. a coverage table without the source counts it live
    let mut c = Cells::new();
    let reports_live = ctx
        .coverage
        .iter()
        .find(|cov| cov.source == REPORT_SOURCE)
        .and_then(|cov| cov.is_live(to_naive(t)))
        .unwrap_or(true);
    let latest_period_end = sc.period_ends.iter().next_back().copied();
    if reports_live {
        summary.reports_known = sc.period_ends.len() as u32;
        c.num(
            f("filing.reports_known"),
            Unit::Count,
            sc.period_ends.len() as f32,
        );
    }
    if !reports_live {
        // unknown: no filing state
    } else if let Some(p) = latest_period_end {
        let pn = to_naive(p);
        summary.fye_month = Some(pn.month());
        c.cat(
            f("filing.fye_month"),
            code("filing.fye_month", &pn.month().to_string()),
        );
        c.days(f("calendar.days_since_fye"), (t - p) as f32);
        // expected period ends after the latest known one, their deadlines, the first missing
        let mut k = 1u32;
        let mut next_deadline: Option<Day> = None;
        let mut overdue_since: Option<Day> = None;
        loop {
            let pk = from_naive(add_months(pn, 12 * k));
            let deadline = from_naive(add_months(to_naive(pk), FILING_DEADLINE_MONTHS));
            if deadline > t {
                next_deadline = Some(deadline);
                break;
            }
            let covered = sc
                .period_ends
                .range(pk - PERIOD_MATCH_DAYS..=pk + PERIOD_MATCH_DAYS)
                .next()
                .is_some();
            if !covered && overdue_since.is_none() {
                overdue_since = Some(deadline);
            }
            k += 1;
            if k > 60 {
                break;
            }
        }
        if let Some(d) = next_deadline {
            c.days(f("calendar.days_to_deadline"), (d - t) as f32);
            c.months(f("filing.months_to_deadline"), months_between(t, d) as f32);
        }
        summary.overdue = overdue_since.is_some();
        c.flag(f("filing.overdue"), summary.overdue);
        if let Some(d) = overdue_since {
            c.months(f("filing.overdue_months"), months_between(d, t) as f32);
        }
    } else if let Some(r) = registered {
        // no report ever: the first is due 25 months after registration at the latest
        let first_deadline = from_naive(add_months(to_naive(r), FIRST_REPORT_MONTHS));
        if first_deadline > t {
            c.days(f("calendar.days_to_deadline"), (first_deadline - t) as f32);
            c.months(
                f("filing.months_to_deadline"),
                months_between(t, first_deadline) as f32,
            );
            c.flag(f("filing.overdue"), false);
        } else {
            summary.overdue = true;
            c.flag(f("filing.overdue"), true);
            c.months(
                f("filing.overdue_months"),
                months_between(first_deadline, t) as f32,
            );
        }
    }
    c.cat(
        f("calendar.month"),
        code("calendar.month", &to_naive(t).month().to_string()),
    );
    if let Some(r) = latest {
        c.months(f("filing.age_months"), months_between(r.valid, t) as f32);
        let mut streak = 0u32;
        for &i in sc.figures.iter().rev() {
            match fig(&s.records[i as usize], "figure.net_result") {
                Some(v) if v < 0.0 => streak += 1,
                _ => break,
            }
        }
        c.num(f("filing.loss_streak"), Unit::Count, streak as f32);
        let equity = fig(r, "figure.equity");
        let total = fig(r, "figure.total_assets");
        let untaxed = fig(r, "figure.untaxed_reserves").unwrap_or(0.0);
        if let Some(x) = ratio(equity.map(|e| e + UNTAXED_EQUITY_SHARE * untaxed), total) {
            c.num(f("filing.equity_ratio_adj"), Unit::Ratio, x);
        }
        if let Some(x) = ratio(fig(r, "figure.cash"), total) {
            c.num(f("filing.cash_ratio"), Unit::Ratio, x);
        }
        if let Some(x) = ratio(fig(r, "figure.short_term_liabilities"), total) {
            c.num(f("filing.short_debt_ratio"), Unit::Ratio, x);
        }
        let bank = match (
            fig(r, "figure.long_term_bank_debt"),
            fig(r, "figure.short_term_bank_debt"),
        ) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
        };
        if let Some(x) = ratio(bank, total) {
            c.num(f("filing.bank_debt_ratio"), Unit::Ratio, x);
        }
        let ebit = match (
            fig(r, "figure.operating_result"),
            fig(r, "figure.financial_income"),
        ) {
            (Some(o), fi) => Some(o + fi.unwrap_or(0.0)),
            _ => None,
        };
        if let Some(x) = ratio(
            ebit,
            fig(r, "figure.financial_expenses").filter(|e| *e > 0.0),
        ) {
            c.num(
                f("filing.interest_coverage"),
                Unit::Ratio,
                x.clamp(-1000.0, 1000.0),
            );
        }
        if let Some(nr) = fig(r, "figure.net_result") {
            c.flag(f("filing.loss"), nr < 0.0);
        }
        if let Some(e) = equity {
            c.flag(f("filing.negative_equity"), e < 0.0);
            if let Some(sc_) = fig(r, "figure.share_capital").filter(|x| *x > 0.0) {
                c.flag(f("filing.equity_below_half_capital"), e < 0.5 * sc_);
            }
        }
    }
    out.push(
        Record {
            kind: ids.state_filing,
            ..base
        },
        &c.cells,
    );

    // sector comparison
    if let (Some(r), Some(sni)) = (latest, summary.sni.as_deref())
        && let Some(q) = ctx.sector.view(t).lookup(sni, summary.employees)
    {
        let band = BAND_LABELS[usize::from(band_for(summary.employees))];
        for (i, name) in FIGURES.iter().enumerate() {
            let [q1, med, q3] = q[i];
            if !(q1.is_finite() && med.is_finite() && q3.is_finite()) {
                continue;
            }
            let (own_field, factor) = SECTOR_MAP[i];
            let own = match own_field {
                Some(field) => fig(r, field).map(|v| v * factor),
                None => ratio(
                    fig(r, "figure.net_sales"),
                    summary.employees.filter(|e| *e > 0.0),
                )
                .map(|v| v / 1000.0),
            };
            let mut c = Cells::new();
            c.cat(f("sector.figure"), code("sector.figure", name));
            c.num(f("sector.q1"), Unit::None, q1);
            c.num(f("sector.median"), Unit::None, med);
            c.num(f("sector.q3"), Unit::None, q3);
            c.cat(f("sector.band"), code("sector.band", band));
            if let Some(o) = own {
                c.num(f("sector.own"), Unit::None, o);
                c.num(f("sector.diff"), Unit::None, o - med);
                let pos = if o < q1 {
                    "below_q1"
                } else if o > q3 {
                    "above_q3"
                } else {
                    "within"
                };
                c.cat(f("sector.position"), code("sector.position", pos));
            }
            out.push(
                Record {
                    kind: ids.state_sector,
                    ..base
                },
                &c.cells,
            );
            summary.sector_records += 1;
        }
    }

    // coverage
    let tn = to_naive(t);
    for cov in &ctx.coverage {
        let mut c = Cells::new();
        c.cat(f("coverage.source"), code("coverage.source", &cov.source));
        let state = match cov.is_live(tn) {
            Some(true) => "live",
            Some(false) if cov.ended.is_some_and(|e| tn >= e) => "ended",
            Some(false) => "not_yet",
            None => "undated",
        };
        c.cat(f("coverage.state"), code("coverage.state", state));
        if let Some(from) = cov.live_from
            && tn >= from
        {
            c.days(f("coverage.days_live"), (t - from_naive(from)) as f32);
        }
        out.push(
            Record {
                kind: ids.state_coverage,
                ..base
            },
            &c.cells,
        );
    }
    let _ = DAY_MAX;
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_and_ratios() {
        assert_eq!(size_band(None), "unknown");
        assert_eq!(size_band(Some(0.0)), "0");
        assert_eq!(size_band(Some(3.0)), "1-4");
        assert_eq!(size_band(Some(7.0)), "5-9");
        assert_eq!(size_band(Some(1200.0)), "500+");
        assert_eq!(ratio(Some(1.0), Some(0.0)), None);
        assert_eq!(ratio(Some(1.0), Some(4.0)), Some(0.25));
    }
}
