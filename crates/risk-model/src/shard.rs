//! the shard file: magic `TICSHD01`, a u32 header length, a JSON header naming every array with
//! its dtype, shape and byte offset, then the arrays, each 8-byte aligned. numpy reads the header
//! and memmaps the arrays. one shard = one batch of `rows` rows of `seq_len` tokens.

use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Error;

pub const MAGIC: &[u8; 8] = b"TICSHD01";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayInfo {
    pub name: String,
    /// numpy dtype string: `<u2`, `<i4`, `<f4`, `|u1`, `<u4`, `<i2`
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub version: u32,
    pub seq_len: usize,
    pub rows: usize,
    pub samples: usize,
    pub cells: usize,
    pub targets: usize,
    pub target_cells: usize,
    pub vocab_size: usize,
    pub vocab_hash: String,
    pub stats_cut: String,
    pub corpora: Vec<String>,
    pub corpus: u16,
    pub anchors: String,
    pub split: String,
    pub seed: u64,
    pub arrays: Vec<ArrayInfo>,
}

/// a typed column being written
pub enum Column {
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
}

impl Column {
    fn dtype(&self) -> &'static str {
        match self {
            Column::U8(_) => "|u1",
            Column::U16(_) => "<u2",
            Column::I16(_) => "<i2",
            Column::U32(_) => "<u4",
            Column::I32(_) => "<i4",
            Column::F32(_) => "<f4",
        }
    }

    fn len(&self) -> usize {
        match self {
            Column::U8(v) => v.len(),
            Column::U16(v) => v.len(),
            Column::I16(v) => v.len(),
            Column::U32(v) => v.len(),
            Column::I32(v) => v.len(),
            Column::F32(v) => v.len(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Column::U8(v) => v.clone(),
            Column::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Column::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Column::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Column::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Column::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

fn align8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// writes a shard: `arrays` as (name, column, shape); the shape's product must equal the length
pub fn write(
    path: &Path,
    mut header: Header,
    arrays: Vec<(&str, Column, Vec<usize>)>,
) -> Result<(), Error> {
    let mut infos = Vec::with_capacity(arrays.len());
    let mut offset = 0usize;
    let mut blobs = Vec::with_capacity(arrays.len());
    for (name, col, shape) in &arrays {
        let n: usize = shape.iter().product();
        if n != col.len() {
            return Err(Error::Shard(
                path.display().to_string(),
                format!("{name}: shape {shape:?} vs length {}", col.len()),
            ));
        }
        let bytes = col.bytes();
        infos.push(ArrayInfo {
            name: (*name).to_owned(),
            dtype: col.dtype().to_owned(),
            shape: shape.clone(),
            offset,
            bytes: bytes.len(),
        });
        offset += align8(bytes.len());
        blobs.push(bytes);
    }
    header.arrays = infos;
    let hj = serde_json::to_vec(&header)?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    out.write_all(MAGIC)?;
    out.write_all(&(hj.len() as u32).to_le_bytes())?;
    out.write_all(&hj)?;
    let pad = align8(12 + hj.len()) - (12 + hj.len());
    out.write_all(&vec![0u8; pad])?;
    for b in &blobs {
        out.write_all(b)?;
        let pad = align8(b.len()) - b.len();
        out.write_all(&vec![0u8; pad])?;
    }
    out.flush()?;
    Ok(())
}

/// a shard read back whole: header plus the raw array bytes
pub struct Shard {
    pub header: Header,
    pub data: Vec<u8>,
}

impl Shard {
    pub fn read(path: &Path) -> Result<Shard, Error> {
        let bad = |m: &str| Error::Shard(path.display().to_string(), m.to_owned());
        let mut f = std::fs::File::open(path)?;
        let mut all = Vec::new();
        f.read_to_end(&mut all)?;
        if all.len() < 12 || &all[..8] != MAGIC {
            return Err(bad("wrong magic"));
        }
        let hl = u32::from_le_bytes([all[8], all[9], all[10], all[11]]) as usize;
        if all.len() < 12 + hl {
            return Err(bad("truncated header"));
        }
        let header: Header = serde_json::from_slice(&all[12..12 + hl])?;
        if header.version != VERSION {
            return Err(bad("wrong version"));
        }
        let start = align8(12 + hl);
        Ok(Shard {
            header,
            data: all[start..].to_vec(),
        })
    }

    fn info(&self, name: &str) -> Result<&ArrayInfo, Error> {
        self.header
            .arrays
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| Error::Shard(name.to_owned(), "missing array".into()))
    }

    fn raw(&self, name: &str) -> Result<&[u8], Error> {
        let a = self.info(name)?;
        self.data
            .get(a.offset..a.offset + a.bytes)
            .ok_or_else(|| Error::Shard(name.to_owned(), "array out of range".into()))
    }

    pub fn u8(&self, name: &str) -> Result<Vec<u8>, Error> {
        Ok(self.raw(name)?.to_vec())
    }

    pub fn u16(&self, name: &str) -> Result<Vec<u16>, Error> {
        Ok(self
            .raw(name)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    }

    pub fn i16(&self, name: &str) -> Result<Vec<i16>, Error> {
        Ok(self
            .raw(name)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect())
    }

    pub fn u32(&self, name: &str) -> Result<Vec<u32>, Error> {
        Ok(self
            .raw(name)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }

    pub fn i32(&self, name: &str) -> Result<Vec<i32>, Error> {
        Ok(self
            .raw(name)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        Ok(self
            .raw(name)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("risk-model-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_default();
        let path = dir.join("s.bin");
        let header = Header {
            version: VERSION,
            seq_len: 4,
            rows: 2,
            samples: 1,
            cells: 3,
            targets: 0,
            target_cells: 0,
            vocab_size: 10,
            vocab_hash: "x".into(),
            stats_cut: "2025-01-01".into(),
            corpora: vec!["t".into()],
            corpus: 0,
            anchors: "a".into(),
            split: "train".into(),
            seed: 1,
            arrays: vec![],
        };
        write(
            &path,
            header,
            vec![
                (
                    "tok_kind",
                    Column::U16(vec![1, 2, 3, 0, 5, 6, 7, 8]),
                    vec![2, 4],
                ),
                ("cell_value", Column::F32(vec![0.5, -1.0, 2.0]), vec![3]),
                (
                    "tok_sample",
                    Column::I16(vec![0, 0, 0, -1, 0, 0, 0, 0]),
                    vec![2, 4],
                ),
            ],
        )
        .unwrap_or_default();
        let s = Shard::read(&path).ok();
        let s = s.as_ref();
        assert!(s.is_some());
        if let Some(s) = s {
            assert_eq!(s.header.rows, 2);
            assert_eq!(
                s.u16("tok_kind").unwrap_or_default(),
                vec![1, 2, 3, 0, 5, 6, 7, 8]
            );
            assert_eq!(
                s.f32("cell_value").unwrap_or_default(),
                vec![0.5, -1.0, 2.0]
            );
            assert_eq!(s.i16("tok_sample").unwrap_or_default()[3], -1);
            assert_eq!(s.header.arrays[1].offset, 16);
        }
        std::fs::remove_dir_all(&dir).unwrap_or_default();
    }
}
