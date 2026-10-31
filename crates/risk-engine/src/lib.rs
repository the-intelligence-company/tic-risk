//! the TIC Risk Open Model's engine: the trained network run on the CPU (`cpu.rs`) from the
//! release's files unchanged; the view the model reads, built from a company's facts as lines
//! (`view.rs`); the input from files (`input.rs`); the answer with its calibration and band
//! (`answer.rs`) and what moves it (`explain.rs`). no database, no network.

pub mod answer;
pub mod artefacts;
pub mod batch;
pub mod calibration;
pub mod cpu;
pub mod error;
pub mod explain;
pub mod forward;
pub mod input;
pub mod live;
pub mod npy;
pub mod outputs;
pub mod view;

pub use error::Error;
