//! the calibration map and the conformal bands fitted per question, applied as written: an
//! isotonic step function in
//! blocks, a logit shift to the anchor's base rate (point in time) or a segment's long-run rate
//! (through the cycle), and a half-width per decile of the calibrated probability (the bands are
//! fitted on pred_cal, so they are looked up and centred on the calibrated value, never the raw one).

use std::path::Path;

use serde::Deserialize;

use crate::error::Result;

#[derive(Debug, Clone, Deserialize)]
pub struct Calibrator {
    pub blocks_x: Vec<f64>,
    pub blocks_y: Vec<f64>,
    pub anchor_shift: f64,
    #[serde(default)]
    pub segment_shift: std::collections::HashMap<String, f64>,
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Calibrator {
    pub fn load(path: &Path) -> Result<Calibrator> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    /// the isotonic step: the first block whose upper edge is ≥ p (searchsorted left)
    pub fn iso(&self, p: f64) -> f64 {
        let idx = self.blocks_x.partition_point(|&x| x < p).min(self.blocks_y.len().saturating_sub(1));
        self.blocks_y.get(idx).copied().unwrap_or(p).clamp(1e-6, 1.0 - 1e-6)
    }

    /// point in time: the isotonic value shifted to the anchor months' base rate
    pub fn pit(&self, p: f64) -> f64 {
        sigmoid(logit(self.iso(p)) + self.anchor_shift)
    }

    /// through the cycle: the segment's long-run shift, the anchor shift where none was fitted
    pub fn ttc(&self, p: f64, segment: &str) -> f64 {
        let shift = self.segment_shift.get(segment).copied().unwrap_or(self.anchor_shift);
        sigmoid(logit(self.iso(p)) + shift)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Bands {
    pub alpha: f64,
    pub edges: Vec<f64>,
    pub q: Vec<f64>,
}

impl Bands {
    pub fn load(path: &Path) -> Result<Bands> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    /// (lower, upper) of the raw probability's band: the decile's half-width (searchsorted right)
    pub fn apply(&self, p: f64) -> (f64, f64) {
        let idx = self.edges.partition_point(|&e| e <= p).saturating_sub(1).min(self.q.len().saturating_sub(1));
        let half = self.q.get(idx).copied().unwrap_or(0.0);
        ((p - half).clamp(0.0, 1.0), (p + half).clamp(0.0, 1.0))
    }
}

/// the risk class 1–5 from the 12-month point-in-time PD, 1 the highest risk
pub fn risk_class(pd_12m_pit: f64) -> u8 {
    if pd_12m_pit < 0.003 {
        5
    } else if pd_12m_pit < 0.008 {
        4
    } else if pd_12m_pit < 0.02 {
        3
    } else if pd_12m_pit < 0.05 {
        2
    } else {
        1
    }
}

/// the score 0–100 as scaled 12-month log-odds, 100 = safest; anchored so that the class
/// boundaries fall on round scores (0.3 % → 80, 5 % → 20)
pub fn score(pd_12m_pit: f64) -> u8 {
    let lo = logit(pd_12m_pit.clamp(1e-6, 1.0 - 1e-6));
    let (a, b) = (logit(0.003), logit(0.05)); // 80 and 20
    let s = 20.0 + (lo - b) / (a - b) * 60.0;
    s.round().clamp(0.0, 100.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_steps_and_bands() {
        let c = Calibrator { blocks_x: vec![0.001, 0.01, 0.1], blocks_y: vec![0.002, 0.02, 0.2], anchor_shift: 0.0, segment_shift: Default::default() };
        assert!((c.iso(0.0005) - 0.002).abs() < 1e-12);
        assert!((c.iso(0.005) - 0.02).abs() < 1e-12);
        assert!((c.iso(0.5) - 0.2).abs() < 1e-12);
        let b = Bands { alpha: 0.1, edges: vec![0.0, 0.01, 0.1, 1.0], q: vec![0.001, 0.01, 0.1] };
        let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12;
        assert!(near(b.apply(0.05), (0.04, 0.06)), "{:?}", b.apply(0.05));
        assert!(near(b.apply(0.5), (0.4, 0.6)), "{:?}", b.apply(0.5));
        assert_eq!(risk_class(0.001), 5);
        assert_eq!(risk_class(0.03), 2);
        assert_eq!(score(0.003), 80);
        assert_eq!(score(0.05), 20);
    }
}
