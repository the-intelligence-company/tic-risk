//! the terminal report of `tic-risk score`: the headline question with its class, every question's
//! probability with its band, and the answers over the as-of days as a chart. drawn with ratatui
//! into a buffer and printed line by line, so it scrolls like any output and works in any terminal
//! (crossterm writes the colours on Windows and Linux alike); without colour when piped

use std::io::{self, Write};

use chrono::{Datelike, NaiveDate};
use ratatui::backend::IntoCrossterm;
use ratatui::buffer::Buffer;
use ratatui::crossterm::queue;
use ratatui::crossterm::style::{ContentStyle, Print, PrintStyledContent};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine, Points};
use ratatui::widgets::{Axis, Block, BorderType, Chart, Dataset, GraphType, Paragraph, Row, Table, Widget};
use risk_engine::answer::Answer;
use risk_engine::explain::Effect;

use crate::check::{Check, Level};

/// one as-of day's answers, in the order asked; an error is the answer's message
pub struct Day {
    pub as_of: NaiveDate,
    pub answers: Vec<(&'static str, Result<Answer, String>)>,
}

/// what moves one question on one day: the full answer and each group's effect
pub struct Explained<'a> {
    pub as_of: NaiveDate,
    pub base: &'a Answer,
    pub effects: &'a [Effect],
}

/// the month-by-month chart: the 12-month bankruptcy probability back from the latest as-of day,
/// and the company's events by the month they became known
pub struct Timeline {
    pub as_of: NaiveDate,
    pub months_back: i32,
    /// (month offset, ≤ 0; the calibrated probability) per month scored
    pub path: Vec<(i32, f64)>,
    /// (month offset, row) per event
    pub events: Vec<(i32, &'static str)>,
}

/// the event strip's rows, top to bottom: key, label, the kinds each counts
pub const ROWS: [(&str, &str, &[&str]); 10] = [
    ("board_owners_officers", "Board and owners", &["role.board_member", "role.deputy_board_member", "role.chair", "role.ceo", "role.deputy_ceo", "role.other", "role.auditor", "ownership.beneficial", "ownership.beneficial_via", "board.composition"]),
    ("address_name_industry", "Name, industry", &["address.registered", "industry.code", "name.changed", "purpose.changed", "articles.changed"]),
    ("annual_report", "Annual report", &["report.registered", "report.received", "report.approved"]),
    ("findings_registry_cases", "Findings, registry cases", &["report.finding", "report.shortage", "case.received", "case.terminated"]),
    ("late_fees_surcharges", "Late fees", &["fee.registered", "tax.late_fee_return", "tax.late_fee_income", "tax.surcharge"]),
    ("payroll_corrected", "Payroll corrections", &["payroll.corrected"]),
    ("tax_registration", "Tax registration", &["tax.registration"]),
    ("enforcement_payment_orders", "Payment orders, debt", &["debt.registered", "debt.present", "debt.changed", "enforcement.case_registered", "enforcement.no_assets_report", "payment_order.applied", "payment_order.verdict", "payment_order.dismissed", "payment_order.withdrawn", "payment_order.changed", "payment_order.removed"]),
    ("board_gap_liquidation_order", "Board vacancy, liquidation order", &["board.gap_started", "liquidation_order.issued"]),
    ("exit", "Bankruptcy, liquidation", &["bankruptcy.initiated", "liquidation.decided", "reconstruction.decided", "status.struck_off"]),
];

/// the strip row an event kind belongs to, if any
pub fn row_of(kind: &str) -> Option<&'static str> {
    ROWS.iter().find(|(_, _, kinds)| kinds.contains(&kind)).map(|(key, _, _)| *key)
}

fn row_colour(row: &str) -> Color {
    match row {
        "annual_report" => Color::Rgb(66, 165, 245),
        "findings_registry_cases" | "late_fees_surcharges" => Color::Rgb(255, 167, 38),
        "payroll_corrected" => Color::Rgb(171, 71, 188),
        "tax_registration" => Color::Rgb(253, 216, 53),
        "enforcement_payment_orders" | "board_gap_liquidation_order" => Color::Rgb(239, 83, 80),
        "exit" => Color::Rgb(207, 32, 32),
        _ => Color::Rgb(150, 150, 160),
    }
}

/// the class of a 12-month probability (calibration.rs's thresholds)
fn class_of(p: f64) -> u8 {
    if p < 0.003 {
        5
    } else if p < 0.008 {
        4
    } else if p < 0.02 {
        3
    } else if p < 0.05 {
        2
    } else {
        1
    }
}

/// a class's colour darkened to a band behind the line
fn band_colour(c: u8) -> Color {
    match class_colour(c) {
        Color::Rgb(r, g, b) => Color::Rgb((f32::from(r) * 0.22) as u8, (f32::from(g) * 0.22) as u8, (f32::from(b) * 0.22) as u8),
        other => other,
    }
}

const TIMELINE_ROWS: u16 = 13;
/// the chart's probability axis: log10 of the percentage, 0.01 % to 100 %
const LOG_LO: f64 = -2.0;
const LOG_HI: f64 = 2.0;

impl Timeline {
    fn rows(&self) -> Vec<(&'static str, &'static str)> {
        ROWS.iter().filter(|(key, _, _)| self.events.iter().any(|(_, r)| r == key)).map(|(key, label, _)| (*key, *label)).collect()
    }

    fn height(&self) -> u16 {
        1 + 1 + TIMELINE_ROWS + 1 + self.rows().len() as u16 + 1 + 1
    }

    fn draw(&self, area: Rect, buf: &mut Buffer) {
        let rows = self.rows();
        let [_, title, chart, axis, strip, legend] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(TIMELINE_ROWS), Constraint::Length(1), Constraint::Length(rows.len() as u16), Constraint::Length(1)]).areas(area);
        Paragraph::new(Line::from(format!(" Bankruptcy risk within 12 months · last {} months", self.months_back)).bold()).render(title, buf);
        let gutter = rows.iter().map(|(_, l)| l.chars().count()).max().unwrap_or(0).max(7) as u16 + 2;
        let plot_w = area.width.saturating_sub(gutter + 3).max(10);
        let x0 = area.x + gutter;
        let n = self.months_back.max(1);
        let col = |m: i32| x0 + (((m + n) as f64) * f64::from(plot_w - 1) / f64::from(n as u16)).round() as u16;
        let h = TIMELINE_ROWS;
        // the class bands, and each class's number at the right of its band
        let mut band_rows: Vec<(u8, u16)> = Vec::new();
        for r in 0..h {
            let top = LOG_HI - f64::from(r) * (LOG_HI - LOG_LO) / f64::from(h);
            let bottom = LOG_HI - f64::from(r + 1) * (LOG_HI - LOG_LO) / f64::from(h);
            let class = class_of(10f64.powf((top + bottom) / 2.0) / 100.0);
            band_rows.push((class, r));
            for x in x0..x0 + plot_w {
                buf[(x, chart.y + r)].set_bg(band_colour(class));
            }
        }
        for c in 1..=5u8 {
            let mine: Vec<u16> = band_rows.iter().filter(|(k, _)| *k == c).map(|(_, r)| *r).collect();
            if let Some(&mid) = mine.get(mine.len() / 2) {
                buf.set_string(x0 + plot_w + 1, chart.y + mid, format!("{c}"), Style::new().fg(class_colour(c)).add_modifier(Modifier::BOLD));
            }
        }
        // the probability axis: one label per decade
        for (v, text) in [(2.0, "100 %"), (1.0, "10 %"), (0.0, "1 %"), (-1.0, "0.1 %"), (-2.0, "0.01 %")] {
            let r = (((LOG_HI - v) / (LOG_HI - LOG_LO)) * f64::from(h - 1)).round() as u16;
            let t = format!("{text:>w$} ", w = usize::from(gutter - 1));
            buf.set_string(area.x, chart.y + r, t, Style::new().fg(Color::DarkGray));
        }
        // the line, drawn on its own canvas and laid over the bands dot by dot (the canvas paints
        // its whole background)
        let log = |p: f64| (p * 100.0).max(1e-4).log10().clamp(LOG_LO, LOG_HI);
        let points: Vec<(f64, f64)> = self.path.iter().map(|&(m, p)| (f64::from(m), log(p))).collect();
        let plot = Rect::new(0, 0, plot_w, h);
        let mut canvas_buf = Buffer::empty(plot);
        Canvas::default()
            .marker(Marker::Braille)
            .x_bounds([-f64::from(n), 0.0])
            .y_bounds([LOG_LO, LOG_HI])
            .paint(|ctx| {
                for w in points.windows(2) {
                    ctx.draw(&CanvasLine::new(w[0].0, w[0].1, w[1].0, w[1].1, Color::White));
                }
                if points.len() == 1 {
                    ctx.draw(&Points { coords: &points, color: Color::White });
                }
            })
            .render(plot, &mut canvas_buf);
        for y in 0..h {
            for x in 0..plot_w {
                let sym = canvas_buf[(x, y)].symbol();
                if sym != " " && sym != "\u{2800}" {
                    let cell = &mut buf[(x0 + x, chart.y + y)];
                    cell.set_symbol(sym);
                    cell.set_fg(Color::White);
                }
            }
        }
        // the time axis: each January's year, and the latest month at the right
        let month_at = |m: i32| add_months(self.as_of, m);
        let mut taken = 0u16;
        for m in -n..=0 {
            let d = month_at(m);
            if d.month() == 1 {
                let x = col(m);
                if x >= taken {
                    buf.set_string(x, axis.y, format!("{}", d.year()), Style::new().fg(Color::DarkGray));
                    taken = x + 5;
                }
            }
        }
        let now = self.as_of.format("%b %Y").to_string();
        let nx = (col(0) + 1).saturating_sub(now.len() as u16);
        if nx >= taken {
            buf.set_string(nx, axis.y, now, Style::new().fg(Color::DarkGray));
        }
        // the event strip: a dot per month and row with events
        for (i, (key, label)) in rows.iter().enumerate() {
            let y = strip.y + i as u16;
            buf.set_string(area.x, y, format!("{label:>w$} ", w = usize::from(gutter - 1)), Style::new().fg(Color::DarkGray));
            for x in x0..x0 + plot_w {
                buf.set_string(x, y, "·", Style::new().fg(Color::Rgb(60, 60, 66)));
            }
            for &(m, r) in &self.events {
                if r == *key && (-n..=0).contains(&m) {
                    buf.set_string(col(m), y, "●", Style::new().fg(row_colour(key)));
                }
            }
        }
        let latest = self.path.last().map(|&(_, p)| p);
        let mut spans = vec![Span::raw(" ━ ").fg(Color::White), Span::raw("monthly probability").dark_gray()];
        if let Some(p) = latest {
            spans.push(Span::raw(format!(" · current {} (class {})", pct(p), class_of(p))).dark_gray());
        }
        spans.push(Span::raw(" · dots: events by month").dark_gray());
        Paragraph::new(Line::from(spans)).render(legend, buf);
    }
}

/// the same day `m` months away (the month's last day when it is shorter)
pub fn add_months(d: NaiveDate, m: i32) -> NaiveDate {
    let r = if m >= 0 { d.checked_add_months(chrono::Months::new(m.unsigned_abs())) } else { d.checked_sub_months(chrono::Months::new(m.unsigned_abs())) };
    r.unwrap_or(d)
}

/// whole months from `a` to `b`, by days, floored: an event's place on the time axis
pub fn month_offset(a: NaiveDate, b: NaiveDate) -> i32 {
    let m = (b.year() - a.year()) * 12 + (b.month() as i32 - a.month() as i32);
    let frac = (f64::from(b.day()) - f64::from(a.day())) / 30.0;
    (f64::from(m) + frac).floor() as i32
}

pub struct Report<'a> {
    pub company: i32,
    /// the input's name for the company, shown instead of its number
    pub name: Option<&'a str>,
    pub input: &'a str,
    pub days: &'a [Day],
    pub explained: Option<Explained<'a>>,
    pub timeline: Option<Timeline>,
}

const HEADLINE: &str = "bankruptcy_12m";
const BAR_MIN: u16 = 12;
const CHART_HEIGHT: u16 = 14;

/// the chart key's and the day table's column names
fn short(q: &str) -> &str {
    match q {
        "bankruptcy_12m" => "Bankruptcy 12m",
        "bankruptcy_24m" => "Bankruptcy 24m",
        "enforcement_debt_12m" => "Enforcement debt",
        "fskatt_revoked_12m" => "Tax registration",
        "report_overdue_12m" => "Overdue report",
        other => other,
    }
}

fn label(q: &str) -> &str {
    match q {
        "bankruptcy_12m" => "Bankruptcy within 12 months",
        "bankruptcy_24m" => "Bankruptcy within 24 months",
        "enforcement_debt_12m" => "New or increased enforcement debt, 12 months",
        "fskatt_revoked_12m" => "Tax registration revoked, 12 months",
        "report_overdue_12m" => "Overdue annual report, 12 months",
        other => other,
    }
}

/// one colour per question, the same in the table's bars and the chart's lines
fn colour(q: &str) -> Color {
    match q {
        "bankruptcy_12m" => Color::Rgb(239, 83, 80),
        "bankruptcy_24m" => Color::Rgb(255, 167, 38),
        "enforcement_debt_12m" => Color::Rgb(66, 165, 245),
        "fskatt_revoked_12m" => Color::Rgb(171, 71, 188),
        _ => Color::Rgb(38, 166, 154),
    }
}

/// class 1 is the highest risk (calibration.rs: 5 % and up), 5 the lowest (under 0.3 %)
fn class_colour(c: u8) -> Color {
    match c {
        1 => Color::Rgb(229, 57, 53),
        2 => Color::Rgb(251, 140, 0),
        3 => Color::Rgb(253, 216, 53),
        4 => Color::Rgb(124, 179, 66),
        _ => Color::Rgb(67, 160, 71),
    }
}

fn pct(p: f64) -> String {
    if p > 0.0 && p < 0.001 { "<0.1 %".to_owned() } else { format!("{:.1} %", p * 100.0) }
}

/// the probability shown: calibrated when the model has the question's calibration, else raw
fn shown(a: &Answer) -> f64 {
    a.p_pit.unwrap_or(a.p_raw)
}

/// the probability as a bar over 0–100 %: solid to the probability (eighth blocks at the edge, a
/// tick for a probability too small for one), shaded on to the band's upper end
fn bar(p: f64, band: Option<[f64; 2]>, width: usize, colour: Color) -> Line<'static> {
    const EIGHTHS: [char; 7] = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let cells = p.clamp(0.0, 1.0) * width as f64;
    let full = (cells.floor() as usize).min(width);
    let frac = ((cells - full as f64) * 8.0).round() as usize;
    let mut solid = "█".repeat(full);
    let mut used = full;
    if used < width && frac > 0 {
        solid.push(if frac >= 8 { '█' } else { EIGHTHS[frac - 1] });
        used += 1;
    }
    if used == 0 && p > 0.0 {
        solid.push('▏');
        used = 1;
    }
    let upper = band.map_or(0, |b| ((b[1].clamp(0.0, 1.0) * width as f64).ceil() as usize).min(width));
    let shade = upper.saturating_sub(used);
    Line::from(vec![
        Span::styled(solid, Style::new().fg(colour)),
        Span::styled("░".repeat(shade), Style::new().fg(Color::DarkGray)),
        Span::raw(" ".repeat(width - used - shade)),
    ])
}

/// a round upper bound for the chart's probability axis
fn axis_top(max: f64) -> f64 {
    let steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0];
    steps.into_iter().find(|&s| s >= max * 1.1).unwrap_or(1.0)
}

impl Report<'_> {
    fn latest(&self) -> Option<&Day> {
        self.days.last()
    }

    fn headline(&self) -> Option<&Answer> {
        let day = self.latest()?;
        day.answers.iter().find(|(q, _)| *q == HEADLINE).or_else(|| day.answers.first()).and_then(|(_, a)| a.as_ref().ok())
    }

    /// the parts' heights: title, headline, the questions' table, and with more than one as-of day
    /// the chart with its key and the table by day; the footer
    fn heights(&self) -> [u16; 8] {
        let rows = self.latest().map_or(0, |d| d.answers.len()) as u16;
        let timeline = self.timeline.as_ref().map_or(0, Timeline::height);
        let explain = self.explained.as_ref().map_or(0, |e| if e.effects.is_empty() { 0 } else { 1 + 3 + e.effects.len() as u16 + 1 });
        let (chart, by_day) = if self.days.len() > 1 { (1 + CHART_HEIGHT + 1, 1 + 3 + self.days.len() as u16) } else { (0, 0) };
        [2, 7, timeline, 3 + rows, explain, chart, by_day, 3]
    }

    /// the report drawn into a buffer `width` columns wide
    pub fn draw(&self, width: u16) -> Buffer {
        let h = self.heights();
        let area = Rect::new(0, 0, width, h.iter().sum());
        let mut buf = Buffer::empty(area);
        let [title, head, timeline, table, explain, chart_area, by_day, foot] = Layout::vertical(h.map(Constraint::Length)).areas(area);
        self.draw_title(title, &mut buf);
        let [head, _] = Layout::vertical([Constraint::Length(6), Constraint::Length(1)]).areas(head);
        self.draw_headline(head, &mut buf);
        if let Some(t) = &self.timeline {
            t.draw(timeline, &mut buf);
        }
        self.draw_table(table, &mut buf);
        if let Some(e) = &self.explained
            && !e.effects.is_empty()
        {
            draw_explained(e, explain, &mut buf);
        }
        if self.days.len() > 1 {
            self.draw_chart(chart_area, &mut buf);
            self.draw_days(by_day, &mut buf);
        }
        let [_, foot] = Layout::vertical([Constraint::Length(1), Constraint::Length(2)]).areas(foot);
        Paragraph::new(vec![
            Line::from(" Probabilities are calibrated. Ranges are 90% prediction intervals.").dark_gray(),
            Line::from(" Use --json for machine-readable output and --show-input to list the events the model read.").dark_gray(),
        ])
        .render(foot, &mut buf);
        buf
    }

    fn draw_title(&self, area: Rect, buf: &mut Buffer) {
        let days = match (self.days.first(), self.days.last()) {
            (Some(a), Some(b)) if self.days.len() > 1 => format!("as of {} to {} ({} dates)", a.as_of, b.as_of, self.days.len()),
            (_, Some(b)) => format!("as of {}", b.as_of),
            _ => String::new(),
        };
        Paragraph::new(vec![
            Line::from(vec![
                Span::raw(" TIC Risk Open Model").bold(),
                Span::raw(format!("   {} · {days}", self.name.map_or_else(|| format!("company {}", self.company), str::to_owned))).dark_gray(),
            ]),
            Line::from(format!(" input: {}", self.input)).dark_gray(),
        ])
        .render(area, buf);
    }

    fn draw_headline(&self, area: Rect, buf: &mut Buffer) {
        let Some(a) = self.headline() else {
            Paragraph::new(" No result").render(area, buf);
            return;
        };
        let block = Block::bordered().border_type(BorderType::Rounded).title(format!(" {} ", label(a.question))).border_style(Style::new().dark_gray());
        let p = shown(a);
        let accent = a.risk_class.map_or(colour(a.question), class_colour);
        let mut first = vec![Span::styled(format!(" {} ", pct(p)), Style::new().fg(accent).add_modifier(Modifier::BOLD))];
        if let (Some(c), Some(s)) = (a.risk_class, a.score) {
            first.push(Span::raw(format!("   risk class {c} of 5   score {s} of 100")));
        }
        let mut lines = vec![Line::from(first)];
        if let Some(class) = a.risk_class {
            let mut scale = vec![Span::raw(" ")];
            // the class in brackets as well as in colour, so it reads without colour too
            for c in 1..=5u8 {
                scale.push(if c == class {
                    Span::styled(format!("[{c}]"), Style::new().fg(Color::Black).bg(class_colour(c)).add_modifier(Modifier::BOLD))
                } else {
                    Span::styled(format!(" {c} "), Style::new().fg(class_colour(c)))
                });
            }
            scale.push(Span::raw("   1 = highest risk, 5 = lowest").dark_gray());
            lines.push(Line::from(scale));
        }
        if let Some([lo, hi]) = a.band90 {
            lines.push(Line::from(format!(" 90% interval {} – {}", pct(lo), pct(hi))));
        }
        lines.push(Line::from(format!(" based on {} events from {} {}", a.tokens, a.neighbours + 1, if a.neighbours == 0 { "company" } else { "companies" })).dark_gray());
        Paragraph::new(lines).block(block).render(area, buf);
    }

    fn draw_table(&self, area: Rect, buf: &mut Buffer) {
        let Some(day) = self.latest() else { return };
        let label_w = day.answers.iter().map(|(q, _)| label(q).chars().count()).max().unwrap_or(10) as u16 + 1;
        let fixed = 1 + label_w + 1 + 11 + 1 + 17 + 1;
        let bar_w = area.width.saturating_sub(fixed + 1).max(BAR_MIN);
        let [title, rest] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
        Paragraph::new(Line::from(format!(" All predictions · as of {}", day.as_of)).bold()).render(title, buf);
        let scale = format!("0 %{:>w$}", "100 %", w = bar_w.saturating_sub(3) as usize);
        let header = Row::new(vec![Line::from(" Prediction"), Line::from("Probability").right_aligned(), Line::from("90% interval"), Line::from(scale)]).dark_gray();
        let rows: Vec<Row> = day
            .answers
            .iter()
            .map(|(q, a)| match a {
                Ok(a) => Row::new(vec![
                    Line::from(format!(" {}", label(q))),
                    Line::from(pct(shown(a))).right_aligned().bold(),
                    Line::from(a.band90.map_or_else(|| "—".to_owned(), |[lo, hi]| format!("{} – {}", pct(lo), pct(hi)))).dark_gray(),
                    bar(shown(a), a.band90, bar_w as usize, colour(q)),
                ]),
                Err(e) => Row::new(vec![Line::from(format!(" {}", label(q))), Line::from("—").right_aligned(), Line::from(e.clone()).dark_gray(), Line::from("")]),
            })
            .collect();
        Table::new(rows, [Constraint::Length(label_w), Constraint::Length(11), Constraint::Length(17), Constraint::Length(bar_w)])
            .header(header)
            .column_spacing(1)
            .block(Block::new().borders(ratatui::widgets::Borders::TOP).border_style(Style::new().dark_gray()))
            .render(rest, buf);
    }

    fn draw_chart(&self, area: Rect, buf: &mut Buffer) {
        let first = self.days.first().map_or(0, |d| d.as_of.num_days_from_ce());
        let questions: Vec<&'static str> = self.latest().map(|d| d.answers.iter().map(|(q, _)| *q).collect()).unwrap_or_default();
        let series: Vec<Vec<(f64, f64)>> = questions
            .iter()
            .map(|q| {
                self.days
                    .iter()
                    .filter_map(|d| d.answers.iter().find(|(x, _)| x == q).and_then(|(_, a)| a.as_ref().ok()).map(|a| (f64::from(d.as_of.num_days_from_ce() - first), shown(a) * 100.0)))
                    .collect()
            })
            .collect();
        let max = series.iter().flatten().map(|&(_, y)| y).fold(0.0, f64::max) / 100.0;
        let top = axis_top(max) * 100.0;
        let span = self.days.last().map_or(1, |d| d.as_of.num_days_from_ce() - first).max(1);
        let datasets: Vec<Dataset> = questions
            .iter()
            .zip(&series)
            .map(|(q, s)| Dataset::default().name(label(q).to_owned()).marker(Marker::Braille).graph_type(GraphType::Line).style(Style::new().fg(colour(q))).data(s))
            .collect();
        let dates: Vec<String> = match (self.days.first(), self.days.get(self.days.len() / 2), self.days.last()) {
            (Some(a), Some(m), Some(b)) if self.days.len() > 2 => vec![a.as_of.to_string(), m.as_of.to_string(), b.as_of.to_string()],
            (Some(a), _, Some(b)) => vec![a.as_of.to_string(), b.as_of.to_string()],
            _ => Vec::new(),
        };
        let [_, chart_area, key] = Layout::vertical([Constraint::Length(1), Constraint::Length(CHART_HEIGHT), Constraint::Length(1)]).areas(area);
        let mut spans = vec![Span::raw(" ")];
        for q in &questions {
            spans.push(Span::styled("━ ", Style::new().fg(colour(q))));
            spans.push(Span::raw(format!("{}  ", short(q))));
        }
        Paragraph::new(Line::from(spans)).render(key, buf);
        Chart::new(datasets)
            .block(Block::bordered().border_type(BorderType::Rounded).title(" Over time ").border_style(Style::new().dark_gray()))
            .x_axis(Axis::default().bounds([0.0, f64::from(span)]).labels(dates).style(Style::new().dark_gray()))
            .y_axis(Axis::default().bounds([0.0, top]).labels([pct(0.0), pct(top / 200.0), pct(top / 100.0)]).style(Style::new().dark_gray()))
            .legend_position(None)
            .render(chart_area, buf);
    }

    /// every as-of day's probabilities, one row a day
    fn draw_days(&self, area: Rect, buf: &mut Buffer) {
        let questions: Vec<&'static str> = self.latest().map(|d| d.answers.iter().map(|(q, _)| *q).collect()).unwrap_or_default();
        let [_, title, rest] = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)]).areas(area);
        Paragraph::new(Line::from(" By date").bold()).render(title, buf);
        let mut header = vec![Line::from(" Date")];
        header.extend(questions.iter().map(|q| Line::from(short(q)).right_aligned()));
        let rows: Vec<Row> = self
            .days
            .iter()
            .map(|d| {
                let mut cells = vec![Line::from(format!(" {}", d.as_of))];
                cells.extend(questions.iter().map(|q| match d.answers.iter().find(|(x, _)| x == q).map(|(_, a)| a) {
                    Some(Ok(a)) => Line::from(pct(shown(a))).right_aligned(),
                    _ => Line::from("—").right_aligned().dark_gray(),
                }));
                Row::new(cells)
            })
            .collect();
        let mut widths = vec![Constraint::Length(12)];
        widths.extend(questions.iter().map(|q| Constraint::Length(short(q).chars().count().max(7) as u16)));
        Table::new(rows, widths)
            .header(Row::new(header).dark_gray())
            .column_spacing(2)
            .block(Block::new().borders(ratatui::widgets::Borders::TOP).border_style(Style::new().dark_gray()))
            .render(rest, buf);
    }
}

/// a signed effect as a bar either side of a centre line: up to `half` cells for the largest
fn diverging(v: f64, max: f64, half: usize) -> Line<'static> {
    let n = if max > 0.0 { ((v.abs() / max) * half as f64).round() as usize } else { 0 };
    let n = n.clamp(usize::from(v != 0.0), half);
    let (left, right) = if v > 0.0 { (" ".repeat(half), "█".repeat(n)) } else { (format!("{}{}", " ".repeat(half - n), "█".repeat(n)), String::new()) };
    let colour = if v > 0.0 { Color::Rgb(229, 57, 53) } else { Color::Rgb(67, 160, 71) };
    Line::from(vec![
        Span::styled(left, Style::new().fg(colour)),
        Span::styled("│", Style::new().fg(Color::DarkGray)),
        Span::styled(right.clone(), Style::new().fg(colour)),
        Span::raw(" ".repeat(half - right.chars().count())),
    ])
}

/// what moves the headline question: each group of facts left out in turn, largest effect first
fn draw_explained(e: &Explained, area: Rect, buf: &mut Buffer) {
    const HALF: usize = 12;
    let [_, title, table, note] = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(2 + e.effects.len() as u16), Constraint::Length(1)]).areas(area);
    Paragraph::new(Line::from(format!(" Drivers of {} · as of {}", label(e.base.question).to_lowercase(), e.as_of)).bold()).render(title, buf);
    let max = e.effects.iter().map(|x| x.log_odds.abs()).fold(0.0, f64::max);
    let base = shown(e.base);
    let header = Row::new(vec![
        Line::from(" Event group"),
        Line::from("events").right_aligned(),
        Line::from(format!("{:>w$}│{:<w$}", "lower ", " higher", w = HALF)),
        Line::from("without").right_aligned(),
        Line::from("change").right_aligned(),
    ])
    .dark_gray();
    let rows: Vec<Row> = e
        .effects
        .iter()
        .map(|x| {
            let without = x.p_pit.unwrap_or(x.p_raw);
            let change = (base - without) * 100.0;
            Row::new(vec![
                Line::from(format!(" {}", x.label)),
                Line::from(x.facts.to_string()).right_aligned().dark_gray(),
                diverging(x.log_odds, max, HALF),
                Line::from(pct(without)).right_aligned(),
                Line::from(if change.abs() < 0.05 { "±0.0 pp".to_owned() } else { format!("{change:+.1} pp") }).right_aligned(),
            ])
        })
        .collect();
    let label_w = e.effects.iter().map(|x| x.label.chars().count()).max().unwrap_or(10) as u16 + 1;
    Table::new(rows, [Constraint::Length(label_w), Constraint::Length(6), Constraint::Length(2 * HALF as u16 + 1), Constraint::Length(9), Constraint::Length(9)])
        .header(header)
        .column_spacing(2)
        .block(Block::new().borders(ratatui::widgets::Borders::TOP).border_style(Style::new().dark_gray()))
        .render(table, buf);
    Paragraph::new(Line::from(" Each row shows the probability without that group of events. Red raises the risk, green lowers it.").dark_gray()).render(note, buf);
}

/// the check's findings and the samples per day, as lines
pub fn check_lines(c: &Check) -> Vec<Line<'static>> {
    let mut out = vec![
        Line::from(vec![Span::raw(" tic-risk check").bold(), Span::raw(format!("   {}", c.input)).dark_gray()]),
        Line::from(format!(" company {} · {} events · {} relations · {} {} in lives.csv", c.company, c.events, c.relations, c.companies, if c.companies == 1 { "company" } else { "companies" })).dark_gray(),
        Line::from(""),
    ];
    out.push(if c.ok() {
        Line::from(vec![Span::styled(" ✓ ", Style::new().fg(Color::Rgb(67, 160, 71)).add_modifier(Modifier::BOLD)), Span::raw("Ready to score").bold()])
    } else {
        Line::from(vec![Span::styled(" ✗ ", Style::new().fg(Color::Rgb(229, 57, 53)).add_modifier(Modifier::BOLD)), Span::raw("Cannot be scored").bold()])
    });
    out.push(Line::from(""));
    if c.findings.is_empty() {
        out.push(Line::from("   No problems found").dark_gray());
    }
    for f in &c.findings {
        let (mark, colour) = match f.level {
            Level::Error => ("✗", Color::Rgb(229, 57, 53)),
            Level::Warning => ("!", Color::Rgb(251, 140, 0)),
            Level::Note => ("·", Color::DarkGray),
        };
        out.push(Line::from(vec![Span::styled(format!("   {mark} "), Style::new().fg(colour).add_modifier(Modifier::BOLD)), Span::raw(f.what.clone()), Span::raw(format!("  ({})", f.count)).dark_gray()]));
        for x in &f.examples {
            out.push(Line::from(format!("       {x}")).dark_gray());
        }
    }
    out.push(Line::from(""));
    out.push(Line::from(" What the model will read").bold());
    for d in &c.days {
        out.push(match (&d.error, d.tokens) {
            (Some(e), _) => Line::from(vec![Span::raw(format!("   {}   ", d.as_of)), Span::raw(e.clone()).fg(Color::Rgb(229, 57, 53))]),
            (None, Some(n)) => {
                let related = d.neighbours.unwrap_or(0);
                let mut spans = vec![Span::raw(format!("   {}   {n} events · {related} related {}", d.as_of, if related == 1 { "company" } else { "companies" }))];
                if d.truncated == Some(true) {
                    spans.push(Span::raw(" · oldest events dropped (history too long)").fg(Color::Rgb(251, 140, 0)));
                }
                if let Some(u) = d.unknown_ids.filter(|&u| u > 0) {
                    spans.push(Span::raw(format!(" · {u} unknown codes")).fg(Color::Rgb(251, 140, 0)));
                }
                Line::from(spans)
            }
            _ => Line::from(format!("   {}", d.as_of)),
        });
    }
    out
}

/// lines to stdout as they are (the terminal wraps them); colours through crossterm
pub fn print_text(lines: &[Line], colour: bool) -> io::Result<()> {
    let mut out = io::stdout().lock();
    for l in lines {
        for span in &l.spans {
            flush(&mut out, &span.content, Some(l.style.patch(span.style)), colour)?;
        }
        queue!(out, Print("\n"))?;
    }
    out.flush()
}

/// the buffer's rows to stdout, each ending at its last visible cell; colours through crossterm
pub fn print(buf: &Buffer, colour: bool) -> io::Result<()> {
    let mut out = io::stdout().lock();
    let width = buf.area.width as usize;
    for row in buf.content.chunks(width) {
        let end = row.iter().rposition(|c| c.symbol() != " " || c.bg != Color::Reset).map_or(0, |i| i + 1);
        let mut run = String::new();
        let mut style: Option<Style> = None;
        for c in &row[..end] {
            let s = Style::new().fg(c.fg).bg(c.bg).add_modifier(c.modifier);
            if style != Some(s) {
                flush(&mut out, &run, style, colour)?;
                run.clear();
                style = Some(s);
            }
            run.push_str(c.symbol());
        }
        flush(&mut out, &run, style, colour)?;
        queue!(out, Print("\n"))?;
    }
    out.flush()
}

fn flush(out: &mut impl Write, run: &str, style: Option<Style>, colour: bool) -> io::Result<()> {
    if run.is_empty() {
        return Ok(());
    }
    match style {
        Some(s) if colour => {
            let cs: ContentStyle = s.into_crossterm();
            queue!(out, PrintStyledContent(cs.apply(run)))
        }
        _ => queue!(out, Print(run)),
    }
}
