//! `tic-risk`: the TIC Risk Open Model on the command line. reads
//! a company's input from files and prints the model's answers as a terminal report (`report.rs`),
//! or as JSON lines with --json; logs on stderr. no database, no GPU: the CPU engine; the network
//! only for `download` (`download.rs`). the model directory (checkpoint, model tables, calibration,
//! bands) is where --model or TIC_RISK_MODEL says, else the working directory or the command's own,
//! each also through its `tic-risk-model/`, the directory `download` writes by default
mod check;
mod download;
mod report;

// one allocator on Windows and Linux alike; see Cargo.toml for why
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use risk_engine::answer::{self, Asked, QUESTIONS};
use risk_engine::artefacts::Artefacts;
use risk_engine::calibration::{Bands, Calibrator};
use risk_engine::cpu::CpuEngine;
use risk_engine::explain;
use risk_engine::input::Input;
use risk_engine::live::Live;

#[derive(Parser)]
#[command(name = "tic-risk", version, about = "Run the TIC Risk Open Model on a company's data files")]
struct Cli {
    /// Path to the model directory (default: the current directory or ./tic-risk-model, then the
    /// directory of this binary)
    #[arg(long, global = true, env = "TIC_RISK_MODEL", value_name = "DIR")]
    model: Option<PathBuf>,
    /// Print JSON lines instead of the report
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Score a company
    Score {
        /// Directory with the company's input files
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Date to score as of (YYYY-MM-DD). Can be repeated. Defaults to the dates in meta.json
        #[arg(long, value_name = "DATE")]
        as_of: Vec<NaiveDate>,
        /// Prediction to run: bankruptcy_12m, bankruptcy_24m, enforcement_debt_12m,
        /// fskatt_revoked_12m or report_overdue_12m. Can be repeated. Defaults to all five
        #[arg(long, value_name = "NAME")]
        question: Vec<String>,
        /// Print every event the model reads to stderr
        #[arg(long)]
        show_input: bool,
        /// With --json, add the drivers: one line per event group with its effect
        #[arg(long)]
        explain: bool,
        /// Months of history in the chart (0 hides the chart)
        #[arg(long, default_value_t = 36, value_name = "N")]
        months_back: u32,
    },
    /// Check a company's input files before scoring
    Check {
        /// Directory with the company's input files
        #[arg(long, value_name = "DIR")]
        input: PathBuf,
        /// Date to check as of (YYYY-MM-DD). Can be repeated. Defaults to the dates in meta.json
        #[arg(long, value_name = "DATE")]
        as_of: Vec<NaiveDate>,
    },
    /// Show which model is used and its files
    Info,
    /// Download the model from Hugging Face (about 900 MB)
    Download {
        /// Directory to download the model into
        #[arg(long, value_name = "DIR", default_value = DEFAULT_DIR)]
        dir: PathBuf,
    },
}

const DEFAULT_DIR: &str = "tic-risk-model";

fn holds_model(d: &Path) -> bool {
    d.join("checkpoint").join("model.safetensors").is_file()
}

/// the model directory: --model or TIC_RISK_MODEL, else the working directory, else the command's
/// own directory, each also through `tic-risk-model/`
fn model_root(arg: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(d) = arg {
        anyhow::ensure!(holds_model(&d), "no model found in {} (expected checkpoint/model.safetensors)", d.display());
        return Ok(d);
    }
    let cwd = std::env::current_dir()?;
    let exe = std::env::current_exe()?;
    let here = exe.parent().map(Path::to_path_buf);
    [Some(cwd), here]
        .into_iter()
        .flatten()
        .flat_map(|d| [d.join(DEFAULT_DIR), d])
        .find(|d| holds_model(d))
        .ok_or_else(|| anyhow::anyhow!("no model found. Run `tic-risk download` to get it, or point to it with --model <dir> or TIC_RISK_MODEL"))
}

/// a question with the model's calibration and bands, when it has them
fn model_question(root: &Path, name: &str) -> anyhow::Result<Asked> {
    let q = Live::question(name)?;
    let (c, b) = (root.join("calibration").join(format!("{name}.json")), root.join("bands").join(format!("{name}.json")));
    if !c.exists() {
        tracing::warn!(question = name, path = %c.display(), "no calibration file, showing the uncalibrated probability");
    }
    let cal = if c.exists() { Some(Calibrator::load(&c)?) } else { None };
    let bands = if b.exists() { Some(Bands::load(&b)?) } else { None };
    Ok(Asked { q, cal, bands })
}

/// the report's width: the terminal's, within reason; 100 columns when stdout is not a terminal
fn width() -> u16 {
    if std::io::stdout().is_terminal() {
        ratatui::crossterm::terminal::size().map_or(100, |(w, _)| w.clamp(72, 120))
    } else {
        100
    }
}

/// the cores the engine's matrix products are split over
fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// colour when stdout is a terminal and NO_COLOR is not set
fn colour() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn download_model(dir: &Path) -> anyhow::Result<()> {
    let m = download::Manifest::embedded()?;
    let o = download::Options::from_env();
    eprintln!("Downloading the TIC Risk Open model {} ({:.0} MB) from {}/{} into {}", m.revision, m.total() as f64 / 1e6, o.hub, m.repo, dir.display());
    let s = download::run(&m, dir, &o)?;
    if s.downloaded == 0 {
        eprintln!("The model in {} is already complete.", dir.display());
    } else {
        eprintln!("Done: {} files, {:.0} MB. Every file matches its checksum.", s.downloaded, s.bytes as f64 / 1e6);
    }
    eprintln!("Try it with an example that comes with this tool: tic-risk score --input {}", Path::new("examples").join("company-b").display());
    Ok(())
}

fn corpus(model_dir: &Path) -> anyhow::Result<risk_tokens::vocab::Vocab> {
    let p = model_dir.join("corpus_vocab.json");
    risk_tokens::vocab::Vocab::load(&p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))
}

fn main() -> anyhow::Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt().with_writer(std::io::stderr).with_env_filter(filter).init();
    let cli = Cli::parse();
    if let Cmd::Download { dir } = &cli.cmd {
        return download_model(dir);
    }
    let root = model_root(cli.model.clone())?;
    let (checkpoint, model_dir) = (root.join("checkpoint"), root.join("model"));
    if let Cmd::Check { input, as_of } = &cli.cmd {
        let live = Live::load(&model_dir)?;
        let corpus = corpus(&model_dir)?;
        let inp = Input::read(input)?;
        let days = if as_of.is_empty() { inp.meta.as_of.clone() } else { as_of.clone() };
        let c = check::run(&inp, &input.display().to_string(), &corpus, &live, &days)?;
        if cli.json {
            println!("{}", serde_json::to_string(&c)?);
        } else {
            report::print_text(&report::check_lines(&c), colour())?;
        }
        if !c.ok() {
            std::process::exit(1);
        }
        return Ok(());
    }
    let art = Artefacts::load(&checkpoint, &model_dir)?;
    match &cli.cmd {
        Cmd::Check { .. } | Cmd::Download { .. } => {}
        Cmd::Info => {
            let has = |dir: &str, q: &str| root.join(dir).join(format!("{q}.json")).exists();
            if cli.json {
                let questions: Vec<_> = QUESTIONS.iter().map(|q| serde_json::json!({"question": q, "calibration": has("calibration", q), "bands": has("bands", q)})).collect();
                println!("{}", serde_json::json!({
                    "model": root.display().to_string(), "step": art.manifest.step, "vocabulary": art.manifest.vocab_hash,
                    "threads": threads(), "questions": questions,
                }));
            } else {
                println!("TIC Risk Open Model");
                println!("  model       {}", root.display());
                println!("  weights     step {}, vocabulary {}", art.manifest.step, art.manifest.vocab_hash);
                println!("  CPU threads {}", threads());
                for q in QUESTIONS {
                    let mark = |b: bool| if b { "yes" } else { "no " };
                    println!("  {q:<22} calibration {}  intervals {}", mark(has("calibration", q)), mark(has("bands", q)));
                }
            }
        }
        Cmd::Score { input, as_of, question, show_input, explain, months_back } => {
            let (show_input, explain, months_back) = (*show_input, *explain, *months_back);
            let names: Vec<String> = if question.is_empty() { QUESTIONS.iter().map(|q| (*q).to_owned()).collect() } else { question.clone() };
            let asked = names.iter().map(|n| model_question(&root, n)).collect::<anyhow::Result<Vec<_>>>()?;
            let hash = art.manifest.vocab_hash.clone();
            if !cli.json && std::io::stderr().is_terminal() {
                eprintln!("Loading model…");
            }
            let cpu = CpuEngine::new(art, &checkpoint.join("model.safetensors"))?;
            let engine = &cpu;
            let live = Live::open(&model_dir, &hash)?;
            let corpus = corpus(&model_dir)?;
            let inp = Input::read(input)?;
            let mut days = if as_of.is_empty() { inp.meta.as_of.clone() } else { as_of.clone() };
            days.sort_unstable();
            days.dedup();
            for d in days.iter().filter(|d| !inp.meta.as_of.contains(d)) {
                tracing::warn!(as_of = %d, prepared_for = ?inp.meta.as_of, "this date is not in meta.json, so data on related companies may be missing");
            }
            let context = inp.context()?;
            let (fetched, _) = inp.build(&corpus, &live.md, context.0.clone(), context.1.clone())?;
            let mut out = Vec::with_capacity(days.len());
            for &day in &days {
                let mut answers = Vec::with_capacity(asked.len());
                for a in &asked {
                    let r = answer::answer(engine, &live, &fetched, day, a, true).map(|(ans, sampled)| {
                        if show_input {
                            eprint!("{}", answer::show(&sampled.packed, &live.md.vocab, a.q.name));
                        }
                        ans
                    });
                    if cli.json {
                        match &r {
                            Ok(ans) => {
                                println!("{}", serde_json::to_string(ans)?);
                                if explain {
                                    for e in explain::explain(engine, &live, &inp, &corpus, &context, day, a, ans)? {
                                        let mut line = serde_json::to_value(&e)?;
                                        if let Some(o) = line.as_object_mut() {
                                            o.insert("company".into(), fetched.focal.into());
                                            o.insert("as_of".into(), day.to_string().into());
                                            o.insert("question".into(), a.q.name.into());
                                        }
                                        println!("{line}");
                                    }
                                }
                            }
                            Err(e) => println!("{}", serde_json::json!({"company": fetched.focal, "as_of": day.to_string(), "question": a.q.name, "error": e.to_string()})),
                        }
                    }
                    answers.push((a.q.name, r.map_err(|e| e.to_string())));
                }
                out.push(report::Day { as_of: day, answers });
            }
            if !cli.json {
                // what moves the headline question on the latest day
                let headline = asked.iter().find(|a| a.q.name == "bankruptcy_12m").or(asked.first());
                let latest = out.last().zip(headline).and_then(|(d, a)| d.answers.iter().find(|(q, _)| *q == a.q.name).and_then(|(_, r)| r.as_ref().ok()).map(|base| (d.as_of, a, base)));
                let effects = match latest {
                    Some((day, a, base)) => explain::explain(engine, &live, &inp, &corpus, &context, day, a, base)?,
                    None => Vec::new(),
                };
                let input_name = input.display().to_string();
                let explained = latest.map(|(as_of, _, base)| report::Explained { as_of, base, effects: &effects });
                let timeline = match days.last() {
                    Some(&end) if months_back > 0 => {
                        if std::io::stderr().is_terminal() {
                            eprintln!("Scoring {} months…", months_back + 1);
                        }
                        let a12 = model_question(&root, "bankruptcy_12m")?;
                        let n = months_back as i32;
                        let path = (-n..=0)
                            .filter_map(|m| {
                                let day = report::add_months(end, m);
                                answer::answer(engine, &live, &fetched, day, &a12, true).ok().map(|(a, _)| (m, a.p_pit.unwrap_or(a.p_raw)))
                            })
                            .collect();
                        // the company's own events, by the month they became known
                        let focal = i64::from(inp.meta.focal);
                        let events = inp
                            .events
                            .iter()
                            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                            .filter(|e| e.get("c").and_then(serde_json::Value::as_i64) == Some(focal))
                            .filter_map(|e| {
                                let row = report::row_of(e.get("k")?.as_str()?)?;
                                let known = NaiveDate::parse_from_str(e.get("ka")?.as_str()?, "%Y-%m-%d").ok()?;
                                Some((report::month_offset(end, known), row))
                            })
                            .filter(|(m, _)| (-n..=0).contains(m))
                            .collect();
                        Some(report::Timeline { as_of: end, months_back: n, path, events })
                    }
                    _ => None,
                };
                let r = report::Report { company: fetched.focal, name: inp.meta.name.as_deref(), input: &input_name, days: &out, explained, timeline };
                report::print(&r.draw(width()), colour())?;
            }
        }
    }
    Ok(())
}
