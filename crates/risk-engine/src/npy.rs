//! a reader for the two `.npy` files the artefacts carry: `texts.npy` (float16, [V, 1024]) and
//! `economy_state.npy` (float32, [months, dims]). version 1.0 and 2.0 headers, C order, little
//! endian; nothing else is accepted.

use std::path::Path;

use crate::error::{Error, Result};

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read_f32(path: &Path) -> Result<Npy> {
    let bytes = std::fs::read(path)?;
    let bad = |m: &str| Error::Artefact(path.display().to_string(), m.to_owned());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not an npy file"));
    }
    let (major, header_len, start) = match bytes[6] {
        1 => (1u8, u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => (bytes[6], u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12usize),
        v => return Err(bad(&format!("npy version {v} not supported"))),
    };
    let _ = major;
    let header = std::str::from_utf8(&bytes[start..start + header_len]).map_err(|_| bad("header is not utf-8"))?;
    if header.contains("'fortran_order': True") {
        return Err(bad("fortran order not supported"));
    }
    let descr = field(header, "descr").ok_or_else(|| bad("no descr"))?;
    let shape_s = field(header, "shape").ok_or_else(|| bad("no shape"))?;
    let shape: Vec<usize> = shape_s
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| bad("bad shape")))
        .collect::<Result<_>>()?;
    let n: usize = shape.iter().product();
    let body = &bytes[start + header_len..];
    let data = match descr.trim_matches('\'') {
        "<f4" => {
            if body.len() < n * 4 {
                return Err(bad("truncated"));
            }
            body[..n * 4].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
        }
        "<f2" => {
            if body.len() < n * 2 {
                return Err(bad("truncated"));
            }
            body[..n * 2].as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect()
        }
        other => return Err(bad(&format!("dtype {other} not supported"))),
    };
    Ok(Npy { shape, data })
}

fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("'{key}':");
    let i = header.find(&k)? + k.len();
    let rest = header[i..].trim_start();
    if rest.starts_with('(') {
        let j = rest.find(')')?;
        Some(&rest[..=j])
    } else {
        let j = rest.find(',').unwrap_or(rest.len());
        Some(rest[..j].trim().trim_end_matches('}'))
    }
}
