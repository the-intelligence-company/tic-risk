//! the canonical figure set of a filed annual report: the names and units of the `report.figures`
//! attributes, in attribute order

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// an amount in whole currency units
    Sek,
    /// a fraction (0.25, not 25)
    Ratio,
    /// a ratio not bounded by one (debt to equity)
    Plain,
    /// a count (employees)
    Count,
}

#[derive(Debug, Clone, Copy)]
pub struct Figure {
    pub name: &'static str,
    pub unit: Unit,
}

const fn fig(name: &'static str, unit: Unit) -> Figure {
    Figure { name, unit }
}

use Unit::{Count, Plain, Ratio, Sek};

/// the canonical figure set, in attribute order
pub const FIGURES: &[Figure] = &[
    fig("net_sales", Sek),
    fig("other_operating_income", Sek),
    fig("operating_income", Sek),
    fig("raw_materials", Sek),
    fig("goods_for_resale", Sek),
    fig("other_external_costs", Sek),
    fig("personnel_costs", Sek),
    fig("depreciation", Sek),
    fig("operating_expenses", Sek),
    fig("operating_result", Sek),
    fig("financial_income", Sek),
    fig("financial_expenses", Sek),
    fig("financial_items", Sek),
    fig("result_after_financial", Sek),
    fig("appropriations", Sek),
    fig("result_before_tax", Sek),
    fig("tax", Sek),
    fig("net_result", Sek),
    fig("intangible_assets", Sek),
    fig("tangible_assets", Sek),
    fig("financial_assets", Sek),
    fig("fixed_assets", Sek),
    fig("inventories", Sek),
    fig("accounts_receivable", Sek),
    fig("current_receivables", Sek),
    fig("current_investments", Sek),
    fig("cash", Sek),
    fig("current_assets", Sek),
    fig("total_assets", Sek),
    fig("share_capital", Sek),
    fig("unrestricted_equity", Sek),
    fig("equity", Sek),
    fig("untaxed_reserves", Sek),
    fig("long_term_bank_debt", Sek),
    fig("long_term_liabilities", Sek),
    fig("short_term_bank_debt", Sek),
    fig("accounts_payable", Sek),
    fig("tax_liabilities", Sek),
    fig("short_term_liabilities", Sek),
    fig("total_equity_liabilities", Sek),
    fig("employees", Count),
    fig("floating_charges", Sek),
    fig("property_mortgages", Sek),
    fig("securities", Sek),
    fig("contingent_liabilities", Sek),
    fig("overdraft_granted", Sek),
    fig("dividend", Sek),
    fig("operating_margin", Ratio),
    fig("gross_margin", Ratio),
    fig("net_margin", Ratio),
    fig("roe", Ratio),
    fig("roa", Ratio),
    fig("equity_ratio", Ratio),
    fig("debt_equity", Plain),
    fig("quick_ratio", Ratio),
];
