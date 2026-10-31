//! what each attribute of each record family becomes: a numeric
//! cell in a unit, a categorical code, a flag, a date as days before the record, a period as
//! months, a multi-valued code list, a derived record for a fact that became knowable later, a
//! group key, or nothing. the table is closed; an attribute met in the input but missing here is
//! skipped and counted by the build, so an omission is loud, never silent.

use risk_store::figures::{FIGURES, Unit as FigureUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Unit {
    None = 0,
    Sek = 1,
    Ratio = 2,
    Percent = 3,
    Count = 4,
    Days = 5,
    Months = 6,
    Year = 7,
    Index = 8,
}

impl Unit {
    pub fn code(self) -> &'static str {
        match self {
            Unit::None => "none",
            Unit::Sek => "sek",
            Unit::Ratio => "ratio",
            Unit::Percent => "percent",
            Unit::Count => "count",
            Unit::Days => "days",
            Unit::Months => "months",
            Unit::Year => "year",
            Unit::Index => "index",
        }
    }

    pub fn from_index(i: u8) -> Unit {
        match i {
            1 => Unit::Sek,
            2 => Unit::Ratio,
            3 => Unit::Percent,
            4 => Unit::Count,
            5 => Unit::Days,
            6 => Unit::Months,
            7 => Unit::Year,
            8 => Unit::Index,
            _ => Unit::None,
        }
    }

    /// the scale of the magnitude channel: `asinh(value / scale)`
    pub fn scale(self) -> f32 {
        match self {
            Unit::Sek => 1000.0,
            Unit::Days => 30.0,
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Treat {
    /// a numeric cell in this unit
    Num(Unit),
    /// a categorical code cell
    Cat,
    /// a flag cell (1/0)
    Bool,
    /// a date: days before the record's valid day (negative = ahead)
    Days,
    /// a date set at the time of the record (a deadline): days, negative = ahead
    Deadline,
    /// a `yyyymm` period: months before the record's valid month
    Months,
    /// a declaration period `2021P4`: a code plus the months since that year's end
    Period,
    /// a comma-separated code list: one categorical cell per value
    Multi,
    /// a date after the record's valid day: a derived record of this kind; on or before it, days
    Future(&'static str),
    /// a finding's note: an amount or a date, by content
    Note,
    /// an identifier that keys the same-case group; never a cell
    Group,
    Skip,
}

#[derive(Debug, Clone, Copy)]
pub struct Attr {
    pub family: &'static str,
    pub attr: &'static str,
    /// the field key (`family.attr`, or a renamed one)
    pub field: &'static str,
    pub treat: Treat,
}

const fn a(family: &'static str, attr: &'static str, field: &'static str, treat: Treat) -> Attr {
    Attr {
        family,
        attr,
        field,
        treat,
    }
}

use Treat::{Bool, Cat, Days, Deadline, Future, Group, Months, Multi, Note, Num, Period, Skip};
use Unit::{Count, Percent, Sek};

/// the lag of a derived record: a withdrawal reported in a later file is known the day after
/// that file
pub const DERIVED_LAG_DAYS: i32 = 1;

pub const ATTRS: &[Attr] = &[
    a("bankruptcy", "surplus", "bankruptcy.surplus", Bool),
    a("bankruptcy", "initiated", "bankruptcy.initiated", Days),
    a("liquidation", "decider", "liquidation.decider", Cat),
    a("liquidation", "compulsory", "liquidation.compulsory", Bool),
    a("liquidation", "decided", "liquidation.decided", Days),
    a("reconstruction", "decided", "reconstruction.decided", Days),
    a("status", "status_type", "status.status_type", Cat),
    a("report", "period_start", "report.period_start", Days),
    a("report", "period_end", "report.period_end", Days),
    a("report", "interim", "report.interim", Bool),
    a("report", "consolidated", "report.consolidated", Bool),
    a("report", "ruleset", "report.ruleset", Cat),
    a("report", "arrival", "report.arrival", Days),
    a("report", "code", "report.code", Cat),
    a("report", "respond_by", "report.respond_by", Deadline),
    a("report", "reason", "report.reason", Cat),
    a("report", "months", "report.months", Num(Unit::Months)),
    a("report", "source", "report.source", Cat),
    a("report", "precision", "report.precision", Cat),
    a("report", "remark_code", "report.remark_code", Cat),
    a("report", "auditor_remark", "report.auditor_remark", Bool),
    a("report", "finding", "report.finding", Cat),
    a("report", "score", "report.score", Num(Count)),
    a("report", "note", "report.note", Note),
    a("report", "report", "report.report", Group),
    a("fee", "number", "fee.number", Num(Count)),
    a("fee", "period_end", "fee.period_end", Days),
    a("fee", "legally_binding", "fee.legally_binding", Deadline),
    a("fee", "cause", "fee.cause", Cat),
    a("dividend", "type", "dividend.type", Cat),
    a("dividend", "amount", "dividend.amount", Num(Sek)),
    a("dividend", "currency", "dividend.currency", Cat),
    a("dividend", "period_end", "dividend.period_end", Days),
    a("liquidation_order", "cause", "liquidation_order.cause", Cat),
    a(
        "liquidation_order",
        "reply_by",
        "liquidation_order.reply_by",
        Deadline,
    ),
    a(
        "liquidation_order",
        "liquidation_on",
        "liquidation_order.liquidation_on",
        Deadline,
    ),
    a("case", "codes", "case.code", Multi),
    a("case", "outcome", "case.outcome", Cat),
    a("case", "case_year", "case.case_year", Group),
    a("case", "case_number", "case.case_number", Group),
    a("board", "gap", "board.gap", Cat),
    a("board", "started", "board.started", Days),
    a("board", "members", "board.members", Num(Count)),
    a("board", "deputies", "board.deputies", Num(Count)),
    a("name", "naming_type", "name.naming_type", Cat),
    a("purpose", "length", "purpose.length", Num(Count)),
    a("debt", "case_type", "debt.case_type", Cat),
    a("debt", "debt_type", "debt.debt_type", Cat),
    a("debt", "amount", "debt.amount", Num(Sek)),
    a("debt", "cases", "debt.cases", Num(Count)),
    a("debt", "prev_amount", "debt.prev_amount", Num(Sek)),
    a("debt", "prev_cases", "debt.prev_cases", Num(Count)),
    a("enforcement", "debt_type", "enforcement.debt_type", Cat),
    a("enforcement", "case_type", "enforcement.case_type", Cat),
    a("enforcement", "cause", "enforcement.cause", Cat),
    a(
        "enforcement",
        "original_debt",
        "enforcement.original_debt",
        Num(Sek),
    ),
    a(
        "enforcement",
        "recalled_amount",
        "enforcement.recalled_amount",
        Num(Sek),
    ),
    a("enforcement", "reason", "enforcement.reason", Cat),
    a("enforcement", "case", "enforcement.case", Group),
    a("enforcement", "event_type", "enforcement.event_type", Skip),
    a(
        "payment_order",
        "notification",
        "payment_order.notification",
        Cat,
    ),
    a("payment_order", "amount", "payment_order.amount", Num(Sek)),
    a(
        "payment_order",
        "applicant_is_person",
        "payment_order.applicant_is_person",
        Bool,
    ),
    a(
        "payment_order",
        "applicant_company",
        "payment_order.applicant_company",
        Skip,
    ),
    a("payment_order", "contested", "payment_order.contested", Cat),
    a(
        "payment_order",
        "removal_cause",
        "payment_order.removal_cause",
        Cat,
    ),
    a(
        "payment_order",
        "withdrawn",
        "payment_order.withdrawn",
        Future("payment_order.withdrawn"),
    ),
    a(
        "payment_order",
        "dismissed",
        "payment_order.dismissed",
        Future("payment_order.dismissed"),
    ),
    a(
        "payment_order",
        "to_court",
        "payment_order.to_court",
        Future("payment_order.to_court"),
    ),
    a("payroll", "period", "payroll.period", Months),
    a("payroll", "amount", "payroll.amount", Num(Sek)),
    a("payroll", "prev_amount", "payroll.prev_amount", Num(Sek)),
    a("payroll", "version", "payroll.version", Skip),
    a("tax", "declaration_day", "tax.declaration_day", Days),
    a("tax", "amount", "tax.amount", Num(Sek)),
    a("tax", "period", "tax.period", Period),
    a("tax", "kind", "tax.kind", Cat),
    a("tax", "period_from", "tax.period_from", Months),
    a("tax", "period_to", "tax.period_to", Months),
    a("tax", "month", "tax.month", Months),
    a("tax", "outstanding", "tax.outstanding", Num(Sek)),
    a("tax", "granted", "tax.granted", Num(Sek)),
    a("tax", "withdrawn", "tax.withdrawn", Num(Sek)),
    a("tax", "ceased", "tax.ceased", Num(Sek)),
    a("tax", "fskatt", "tax.fskatt", Cat),
    a("tax", "fskatt_ended", "tax.fskatt_ended", Deadline),
    a("tax", "reasons", "tax.reason", Multi),
    a("tax", "vat", "tax.vat", Bool),
    a("tax", "employer", "tax.employer", Bool),
    a("industry", "sni", "industry.sni", Cat),
    a("industry", "rank", "industry.rank", Num(Count)),
    a("industry", "prob", "industry.prob", Num(Unit::Ratio)),
    a(
        "industry",
        "prob_division",
        "industry.prob_division",
        Num(Unit::Ratio),
    ),
    a("industry", "abstain", "industry.abstain", Bool),
    a("permit", "category", "permit.category", Cat),
    a("permit", "municipality", "permit.municipality", Cat),
    a("permit", "granted", "permit.granted", Days),
    a("role", "code", "role.code", Cat),
    a("role", "auditor_type", "role.auditor_type", Cat),
    a("ownership", "extent", "ownership.extent", Cat),
    a("ownership", "govern", "ownership.govern", Cat),
    a("group", "share", "group.share", Num(Percent)),
    a("address", "type", "address.type", Cat),
    a("address", "municipality", "address.municipality", Cat),
];

/// the family of a kind: the part before the first dot
pub fn family(kind: &str) -> &str {
    kind.split('.').next().unwrap_or(kind)
}

pub fn figure_unit(name: &str) -> Option<Unit> {
    if name == "net_sales_change" || name == "operating_result_change" {
        return Some(Unit::Ratio);
    }
    FIGURES
        .iter()
        .find(|f| f.name == name)
        .map(|f| match f.unit {
            FigureUnit::Sek => Unit::Sek,
            FigureUnit::Ratio | FigureUnit::Plain => Unit::Ratio,
            FigureUnit::Count => Unit::Count,
        })
}

/// the treatment of an attribute of a kind; a figure of `report.figures` is a numeric cell in
/// the figure's unit under `figure.<name>`
pub fn spec(kind: &str, attr: &str) -> Option<(String, Treat)> {
    let fam = family(kind);
    if let Some(a) = ATTRS.iter().find(|a| a.family == fam && a.attr == attr) {
        return Some((a.field.to_owned(), a.treat));
    }
    if kind == "report.figures"
        && let Some(unit) = figure_unit(attr)
    {
        return Some((format!("figure.{attr}"), Num(unit)));
    }
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn specs_resolve() {
        assert_eq!(
            spec("debt.changed", "amount").unwrap(),
            ("debt.amount".to_owned(), Num(Sek))
        );
        assert_eq!(spec("case.received", "codes").unwrap().1, Multi);
        assert_eq!(
            spec("payment_order.applied", "withdrawn").unwrap().1,
            Future("payment_order.withdrawn")
        );
        assert_eq!(
            spec("report.figures", "net_sales").unwrap(),
            ("figure.net_sales".to_owned(), Num(Sek))
        );
        assert_eq!(
            spec("report.figures", "debt_equity").unwrap().1,
            Num(Unit::Ratio)
        );
        assert_eq!(spec("report.figures", "employees").unwrap().1, Num(Count));
        assert_eq!(spec("report.figures", "interim").unwrap().1, Bool);
        assert!(spec("report.figures", "no_such").is_none());
        assert_eq!(family("liquidation_order.issued"), "liquidation_order");
        assert_eq!(Unit::from_index(Unit::Months as u8), Unit::Months);
    }
}
