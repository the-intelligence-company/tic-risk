//! the legal-entity population: one life per company from the company register. registration and
//! ceased dates agree with the state rules: surplus bankruptcies and repealed settlements do not
//! cease a company, the date of a strike-off on own request is unusable. legal forms are a closed
//! enum.

use chrono::NaiveDate;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalForm {
    Aktiebolag,
    Handelsbolag,
    Kommanditbolag,
    EkonomiskForening,
}

impl LegalForm {
    pub const ALL: [LegalForm; 4] = [
        LegalForm::Aktiebolag,
        LegalForm::Handelsbolag,
        LegalForm::Kommanditbolag,
        LegalForm::EkonomiskForening,
    ];

    pub fn from_code(code: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.code() == code)
    }

    pub fn code(self) -> &'static str {
        match self {
            LegalForm::Aktiebolag => "AB",
            LegalForm::Handelsbolag => "HB",
            LegalForm::Kommanditbolag => "KB",
            LegalForm::EkonomiskForening => "EF",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompanyLife {
    pub company_id: i32,
    pub legal_form: LegalForm,
    pub registered: Option<NaiveDate>,
    /// the date the company ended (any cause); None while it lives
    pub ceased: Option<NaiveDate>,
}

impl CompanyLife {
    /// registered on or before `at` and not ended before `at`
    pub fn alive_on(&self, at: NaiveDate) -> bool {
        self.registered.is_some_and(|r| r <= at) && self.ceased.is_none_or(|c| c > at)
    }
}


/// companies alive on `at` (the stock), the denominator of a rate
pub fn stock_on(lives: &[CompanyLife], at: NaiveDate) -> usize {
    lives.iter().filter(|l| l.alive_on(at)).count()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn alive_between_registration_and_cease() {
        let life = CompanyLife {
            company_id: 1,
            legal_form: LegalForm::Aktiebolag,
            registered: Some(d(2000, 5, 1)),
            ceased: Some(d(2010, 1, 1)),
        };
        assert!(!life.alive_on(d(2000, 4, 30)));
        assert!(life.alive_on(d(2000, 5, 1)));
        assert!(life.alive_on(d(2009, 12, 31)));
        assert!(!life.alive_on(d(2010, 1, 1)));
        let unknown_start = CompanyLife {
            registered: None,
            ..life
        };
        assert!(!unknown_start.alive_on(d(2005, 1, 1)));
    }
}
