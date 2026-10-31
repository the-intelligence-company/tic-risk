//! the forward pass's outputs, whichever engine ran it, and the probabilities read from them

pub struct Outputs {
    pub samples: usize,
    pub months: usize,
    pub causes: usize,
    /// [S, months, causes + 1]
    pub surv: Vec<f32>,
    /// [S, months, 3]
    pub early: Vec<f32>,
    /// [S, d_z]
    pub z: Vec<f32>,
    /// bytes of working memory the engine needed at its peak, when it counts them
    pub peak_bytes: usize,
}

/// the early-warning causes by position in the second hazard head
pub fn early_cause(key: &str) -> Option<usize> {
    match key {
        "enforcement_debt" => Some(0),
        "fskatt_revoked" => Some(1),
        "report_overdue" => Some(2),
        _ => None,
    }
}

/// the competing causes a question sums, 1-based
pub fn query_causes(key: &str) -> Vec<usize> {
    match key {
        "bankruptcy" => vec![1],
        "compulsory_liquidation" => vec![2],
        "reconstruction" => vec![3],
        "deregistration" => vec![4],
        _ => vec![1, 2, 3, 4],
    }
}

impl Outputs {
    /// cumulative incidence of one sample: [months, causes]
    pub fn cif(&self, s: usize) -> Vec<f32> {
        let (hm, c1) = (self.months, self.causes + 1);
        let l = &self.surv[s * hm * c1..(s + 1) * hm * c1];
        let mut out = vec![0f32; hm * self.causes];
        let mut log_s_prev = 0f32;
        let mut acc = vec![0f32; self.causes];
        for h in 0..hm {
            let row = &l[h * c1..(h + 1) * c1];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lz = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            let surv_prev = log_s_prev.exp();
            for c in 0..self.causes {
                acc[c] += surv_prev * (row[1 + c] - lz).exp();
                out[h * self.causes + c] = acc[c];
            }
            log_s_prev += row[0] - lz;
        }
        out
    }

    /// the early-warning probabilities of one sample: [months, 3]
    pub fn early_p(&self, s: usize) -> Vec<f32> {
        let hm = self.months;
        let l = &self.early[s * hm * 3..(s + 1) * hm * 3];
        let mut out = vec![0f32; hm * 3];
        for c in 0..3 {
            let mut ls = 0f32;
            for h in 0..hm {
                let x = l[h * 3 + c];
                // log sigmoid(−x)
                ls += if -x >= 0.0 { -(1.0 + (x).exp()).ln() } else { -x - (1.0 + (-x).exp()).ln() };
                out[h * 3 + c] = 1.0 - ls.exp();
            }
        }
        out
    }

    /// the probability of a sample's question within its horizon
    pub fn query_probability(&self, s: usize, key: &str, horizon: usize) -> f32 {
        let h = horizon.clamp(1, self.months) - 1;
        let p = if let Some(c) = early_cause(key) {
            self.early_p(s)[h * 3 + c]
        } else {
            let cif = self.cif(s);
            query_causes(key).iter().map(|&c| cif[h * self.causes + (c - 1)]).sum()
        };
        p.clamp(0.0, 1.0)
    }
}
