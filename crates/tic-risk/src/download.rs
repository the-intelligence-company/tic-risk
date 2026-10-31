//! `tic-risk download`: the model's files from Hugging Face into a directory, each checked against
//! the size and SHA-256 this binary carries (`model_files.txt`), so a binary accepts only the model
//! it was released with. a file already there and correct is kept; an interrupted download resumes
//! where it stopped (`<file>.part`)

use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

const MANIFEST: &str = include_str!("model_files.txt");

/// the hub's address unless HF_ENDPOINT names a mirror (the variable Hugging Face's own tools read)
pub const HUB: &str = "https://huggingface.co";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    pub path: String,
    pub size: u64,
    /// lowercase hex
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub repo: String,
    pub revision: String,
    pub files: Vec<ModelFile>,
}

impl Manifest {
    pub fn embedded() -> anyhow::Result<Manifest> {
        Manifest::parse(MANIFEST)
    }

    /// `repo <id>`, `revision <tag>`, then `<sha256> <size> <path>` per file; `#` starts a comment
    pub fn parse(text: &str) -> anyhow::Result<Manifest> {
        let (mut repo, mut revision, mut files) = (None, None, Vec::new());
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let at = || format!("model_files.txt line {}", n + 1);
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.as_slice() {
                ["repo", r] => repo = Some((*r).to_owned()),
                ["revision", r] => revision = Some((*r).to_owned()),
                [sha, size, path] if sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()) => {
                    // the paths are joined onto the download directory: never absolute, never upwards
                    if path.starts_with('/') || path.contains('\\') || path.split('/').any(|p| p.is_empty() || p == "." || p == "..") {
                        bail!("{}: unsafe path {path}", at());
                    }
                    let size = size.parse().with_context(|| format!("{}: size", at()))?;
                    files.push(ModelFile { path: (*path).to_owned(), size, sha256: sha.to_ascii_lowercase() });
                }
                _ => bail!("{}: not understood", at()),
            }
        }
        let (Some(repo), Some(revision)) = (repo, revision) else {
            bail!("model_files.txt names no repo or revision");
        };
        if files.is_empty() {
            bail!("model_files.txt lists no files");
        }
        Ok(Manifest { repo, revision, files })
    }

    pub fn total(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

pub struct Options {
    pub hub: String,
    pub token: Option<String>,
    pub progress: bool,
}

impl Options {
    pub fn from_env() -> Options {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        Options {
            hub: var("HF_ENDPOINT").unwrap_or_else(|| HUB.to_owned()).trim_end_matches('/').to_owned(),
            token: var("HF_TOKEN"),
            progress: io::stderr().is_terminal(),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub downloaded: usize,
    pub kept: usize,
    pub bytes: u64,
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .user_agent(concat!("tic-risk/", env!("CARGO_PKG_VERSION")))
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(30)))
        .build()
        .into()
}

pub fn sha256_of(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn is_complete(path: &Path, f: &ModelFile) -> io::Result<bool> {
    match fs::metadata(path) {
        Ok(m) if m.is_file() && m.len() == f.size => Ok(sha256_of(path)? == f.sha256),
        Ok(_) => Ok(false),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn part_of(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000.0
}

struct Progress<'a> {
    on: bool,
    path: &'a str,
    size: u64,
    done: u64,
    shown: Instant,
}

impl Progress<'_> {
    fn add(&mut self, n: u64) {
        self.done += n;
        if self.on && self.shown.elapsed() >= Duration::from_millis(200) {
            self.show();
        }
    }

    fn show(&mut self) {
        eprint!("\r  {:<44} {:>8.1} / {:.1} MB", self.path, mb(self.done), mb(self.size));
        self.shown = Instant::now();
    }
}

/// one file into `dst`: `true` when it was downloaded, `false` when it was already there
fn fetch(agent: &ureq::Agent, url: &str, o: &Options, f: &ModelFile, dst: &Path) -> anyhow::Result<bool> {
    if is_complete(dst, f)? {
        return Ok(false);
    }
    if let Some(dir) = dst.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = part_of(dst);
    // a second attempt from the start when the server cannot resume (416)
    for attempt in 0..2 {
        let mut have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have > f.size || attempt > 0 {
            let _ = fs::remove_file(&part);
            have = 0;
        }
        // an empty file is fetched too: there is no part to resume, and it must exist
        if have < f.size || !part.exists() {
            let mut req = agent.get(url);
            if have > 0 {
                req = req.header("Range", &format!("bytes={have}-"));
            }
            if let Some(t) = &o.token {
                req = req.header("Authorization", &format!("Bearer {t}"));
            }
            let mut res = req.call().with_context(|| format!("downloading {}", f.path))?;
            let append = match res.status().as_u16() {
                206 if have > 0 => true,
                200 => false,
                416 => continue,
                401 | 403 => bail!("Hugging Face refused {} ({}). The repository may be private: set HF_TOKEN to a token with access", f.path, res.status()),
                404 => bail!("{} was not found on Hugging Face ({url})", f.path),
                _ => bail!("Hugging Face answered {} for {}", res.status(), f.path),
            };
            let mut out = if append {
                OpenOptions::new().append(true).open(&part)
            } else {
                File::create(&part)
            }
            .with_context(|| format!("writing {}", part.display()))?;
            let mut p = Progress { on: o.progress, path: &f.path, size: f.size, done: if append { have } else { 0 }, shown: Instant::now() };
            let mut body = res.body_mut().as_reader();
            let mut buf = vec![0u8; 1 << 18];
            loop {
                let n = body.read(&mut buf).with_context(|| format!("downloading {}", f.path))?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
                p.add(n as u64);
            }
            out.flush()?;
            drop(out);
            if p.on {
                p.show();
                eprintln!();
            }
        }
        if !is_complete(&part, f)? {
            let _ = fs::remove_file(&part);
            bail!("{} did not match its checksum and was removed. Run the download again", f.path);
        }
        fs::rename(&part, dst).with_context(|| format!("moving {} into place", dst.display()))?;
        return Ok(true);
    }
    bail!("Hugging Face could not resume {}. Run the download again", f.path)
}

/// every file of `m` into `dir`
pub fn run(m: &Manifest, dir: &Path, o: &Options) -> anyhow::Result<Summary> {
    let agent = agent();
    let mut s = Summary::default();
    for f in &m.files {
        let url = format!("{}/{}/resolve/{}/{}", o.hub, m.repo, m.revision, f.path);
        if fetch(&agent, &url, o, f, &dir.join(&f.path))? {
            s.downloaded += 1;
            s.bytes += f.size;
            if !o.progress {
                eprintln!("  {}  {:.1} MB", f.path, mb(f.size));
            }
        } else {
            s.kept += 1;
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    fn sha(b: &[u8]) -> String {
        Sha256::digest(b).iter().map(|b| format!("{b:02x}")).collect()
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tic-risk-download-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// a hub on localhost: `/<repo>/resolve/<rev>/<path>` redirects to `/cdn/<path>`, which serves
    /// the file with Range support; every request line and its Range header are recorded
    fn hub(files: HashMap<String, Vec<u8>>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::new()));
        let (seen, base) = (log.clone(), addr.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut range = None;
                loop {
                    let mut h = String::new();
                    reader.read_line(&mut h).unwrap();
                    if h.trim().is_empty() {
                        break;
                    }
                    if let Some(v) = h.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range = v.trim().trim_end_matches('-').parse::<usize>().ok();
                    }
                }
                let path = first.split_whitespace().nth(1).unwrap_or("").to_owned();
                seen.lock().unwrap().push(format!("{path} {range:?}"));
                let reply = if let Some(name) = path.strip_prefix("/cdn/") {
                    match files.get(name) {
                        Some(body) => match range {
                            Some(r) if r < body.len() => {
                                let head = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {r}-{}/{}\r\nConnection: close\r\n\r\n", body.len() - r, body.len() - 1, body.len());
                                [head.into_bytes(), body[r..].to_vec()].concat()
                            }
                            Some(_) => b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                            None => [format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes(), body.clone()].concat(),
                        },
                        None => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                    }
                } else if let Some(rest) = path.strip_prefix("/org/model/resolve/v1/") {
                    format!("HTTP/1.1 302 Found\r\nLocation: {base}/cdn/{rest}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes()
                } else {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                };
                let _ = stream.write_all(&reply);
            }
        });
        (addr, log)
    }

    fn manifest(files: &[(&str, &[u8])]) -> Manifest {
        let mut text = String::from("# test\nrepo org/model\nrevision v1\n");
        for (p, b) in files {
            text.push_str(&format!("{} {} {p}\n", sha(b), b.len()));
        }
        Manifest::parse(&text).unwrap()
    }

    fn quiet(hub: &str) -> Options {
        Options { hub: hub.to_owned(), token: None, progress: false }
    }

    #[test]
    fn the_embedded_manifest_lists_the_model() {
        let m = Manifest::embedded().unwrap();
        assert_eq!(m.repo, "tic-oss/tic-risk-v2");
        assert!(m.files.iter().any(|f| f.path == "checkpoint/model.safetensors"));
        assert!(m.files.iter().all(|f| !f.path.starts_with("examples/")), "the examples ship with the tool, not the model");
        assert!(m.files.iter().all(|f| !f.path.ends_with("README.md")), "the card can change without a new binary");
    }

    #[test]
    fn a_manifest_refuses_paths_outside_the_directory() {
        let s = "a".repeat(64);
        for bad in ["../x", "/x", "a/../x", "a//x", "a\\x"] {
            assert!(Manifest::parse(&format!("repo r\nrevision v\n{s} 1 {bad}\n")).is_err(), "{bad}");
        }
        assert!(Manifest::parse("repo r\nrevision v\n").is_err());
    }

    #[test]
    fn downloads_through_a_redirect_and_keeps_what_is_there() {
        let big: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        let files = [("checkpoint/model.bin", big.as_slice()), ("a.json", b"{}".as_slice()), ("empty.jsonl", b"".as_slice())];
        let (addr, _) = hub(files.iter().map(|(p, b)| ((*p).to_owned(), b.to_vec())).collect());
        let m = manifest(&files);
        let dir = temp("fresh");
        assert_eq!(run(&m, &dir, &quiet(&addr)).unwrap(), Summary { downloaded: 3, kept: 0, bytes: big.len() as u64 + 2 });
        assert_eq!(fs::read(dir.join("checkpoint/model.bin")).unwrap(), big);
        assert_eq!(fs::metadata(dir.join("empty.jsonl")).unwrap().len(), 0);
        assert_eq!(run(&m, &dir, &quiet(&addr)).unwrap(), Summary { downloaded: 0, kept: 3, bytes: 0 });
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_interrupted_download_resumes_with_a_range() {
        let big: Vec<u8> = (0..1_000_000u32).map(|i| (i % 241) as u8).collect();
        let files = [("model.bin", big.as_slice())];
        let (addr, log) = hub(files.iter().map(|(p, b)| ((*p).to_owned(), b.to_vec())).collect());
        let dir = temp("resume");
        fs::write(dir.join("model.bin.part"), &big[..400_000]).unwrap();
        run(&manifest(&files), &dir, &quiet(&addr)).unwrap();
        assert_eq!(fs::read(dir.join("model.bin")).unwrap(), big);
        assert!(!dir.join("model.bin.part").exists());
        assert!(log.lock().unwrap().iter().any(|l| l == "/cdn/model.bin Some(400000)"), "{:?}", log.lock().unwrap());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_file_that_does_not_match_is_removed_and_reported() {
        let (addr, _) = hub(HashMap::from([("model.bin".to_owned(), b"what the server has".to_vec())]));
        let dir = temp("mismatch");
        let m = manifest(&[("model.bin", b"what the binary expects".as_slice())]);
        let e = run(&m, &dir, &quiet(&addr)).unwrap_err().to_string();
        assert!(e.contains("checksum"), "{e}");
        assert!(!dir.join("model.bin").exists() && !dir.join("model.bin.part").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_file_says_so() {
        let (addr, _) = hub(HashMap::new());
        let dir = temp("missing");
        let e = run(&manifest(&[("gone.bin", b"x".as_slice())]), &dir, &quiet(&addr)).unwrap_err().to_string();
        assert!(e.contains("not found"), "{e}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
